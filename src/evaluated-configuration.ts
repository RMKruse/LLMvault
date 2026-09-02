export const EVALUATED_CONFIGURATION = {
  chatModel: {
    digest: "7c75d6f0f4b974c8761fe291ab4d808754b6a6b1420c60663c519363b49a26e1",
    name: "gemma4:12b-mlx",
  },
  embeddingModel: {
    digest: "df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907",
    name: "qwen3-embedding:4b",
  },
  indexSignature: {
    chunkOverlapBytes: 512,
    chunkTargetBytes: 2_048,
    chunkerVersion: 1,
    extractorVersion: 1,
    schemaVersion: 1,
    vectorDimension: 2_560,
  },
  minimumScore: 0.5710371502360156,
} as const;

export const CALIBRATED_CUTOFFS = [
  {
    embeddingModel: {
      digest: "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d",
      name: "qwen3-embedding:0.6b",
    },
    indexSignature: {
      chunkOverlapBytes: 512,
      chunkTargetBytes: 2_048,
      chunkerVersion: 1,
      extractorVersion: 1,
      schemaVersion: 1,
      vectorDimension: 1_024,
    },
    minimumScore: 0.5448656969693813,
  },
  {
    embeddingModel: EVALUATED_CONFIGURATION.embeddingModel,
    indexSignature: EVALUATED_CONFIGURATION.indexSignature,
    minimumScore: EVALUATED_CONFIGURATION.minimumScore,
  },
] as const;
