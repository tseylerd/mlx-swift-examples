import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

private class Attention: Module {
    let args: Gemma3Configuration
    let scale: Float
    let headDim: Int
    let layerIdx: Int
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    public init(_ args: Gemma3Configuration, layerIdx: Int) {
        self.args = args

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.repeats = args.attentionHeads / args.kvHeads
        self.headDim = args.headDimensions
        self.layerIdx = layerIdx
        self.scale = 1.0 / pow(Float(args.queryPreAttnScalar), 0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        
        self.qNorm = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.kNorm = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.isSliding = (layerIdx + 1) % args.slidingWindowPattern != 0
        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: self.isSliding ? args.ropeLocalBaseFreq : args.ropeGlobalBaseFreq)
    }
    
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }
        var processedMask = mask
        if let mask = processedMask, mask.shape.last! != keys.shape[keys.shape.count - 2] {
            let startIdx = mask.shape.last! - keys.shape[keys.shape.count - 2]
            processedMask = mask[.ellipsis, startIdx...]
        }
        var output = scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: processedMask)
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Gemma3Configuration, layerIdx: Int) {
        self._attention.wrappedValue = Attention(args, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}


public class Gemma3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let model: ModelInner
    private let lmHead: Linear

    public init(_ args: Gemma3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = ModelInner(args)
        self.lmHead = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = lmHead(out)
        return out
    }
}

private class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    fileprivate let norm: RMSNorm
    fileprivate let args: Gemma3Configuration

    let hiddenScale: Float

    public init(_ args: Gemma3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.hiddenScale = pow(Float(args.hiddenSize), 0.5)

        self.layers = (0 ..< args.hiddenLayers)
            .map { layerIdx in
                TransformerBlock(args, layerIdx: layerIdx)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.args = args
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * MLXArray(hiddenScale, dtype: .bfloat16).asType(h.dtype)

        let mask: MLXArray? = createAttentionMask(h: h, cache: cache)
        var fullMask: MLXArray?
        var slidingWindowMask: MLXArray?
        if mask == nil {
            let j = self.args.slidingWindowPattern
            let cacheToPass = cache != nil ? Array(cache![j - 1..<j]) : nil
            fullMask = createAttentionMask(h: h, cache: cacheToPass)
            slidingWindowMask = createAttentionMask(h: h, cache: cacheToPass)
        }
        
        for (i, layer) in layers.enumerated() {
            let isGlobal = i % self.args.slidingWindowPattern == self.args.slidingWindowPattern - 1
            
            var localMask = mask
            if mask == nil && isGlobal {
                localMask = fullMask
            } else if mask == nil {
                localMask = slidingWindowMask
            }
            h = layer(h, mask: localMask, cache: cache?[i])
        }
        
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public struct Gemma3Configuration: Codable {
    var hiddenSize: Int = 1152
    var hiddenLayers: Int = 26
    var intermediateSize: Int = 6912
    var attentionHeads: Int = 4
    var headDimensions: Int = 256
    var rmsNormEps: Float = 1.0e-6
    var vocabularySize: Int = 262144
    var kvHeads: Int = 1
    var ropeGlobalBaseFreq: Float = 1_000_000
    var ropeLocalBaseFreq: Float = 10_000
    var ropeTraditional: Bool = false
    var queryPreAttnScalar: Float = 256
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 6

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys> = try decoder.container(
            keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: CodingKeys.attentionHeads)
        self.headDimensions = try container.decode(
            Int.self, forKey: CodingKeys.headDimensions)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: CodingKeys.kvHeads)
        self.ropeGlobalBaseFreq =
        try container.decodeIfPresent(Float.self, forKey: CodingKeys.ropeGlobalBaseFreq)
            ?? 1_000_000
        self.ropeLocalBaseFreq =
        try container.decodeIfPresent(Float.self, forKey: CodingKeys.ropeLocalBaseFreq)
            ?? 10_000
        self.ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        self.queryPreAttnScalar = try container.decode(
            Float.self, forKey: CodingKeys.queryPreAttnScalar)
        self.slidingWindow = try container.decode(
            Int.self, forKey: CodingKeys.slidingWindow)
        self.slidingWindowPattern = try container.decode(
            Int.self, forKey: CodingKeys.slidingWindowPattern)
    }
}

extension Gemma3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
