export const TIER1 = "Tier 1: Fast Vector Threat Cache (ChromaDB)";
export const TIER2 = "Tier 2: On-Device Intent Guard (Qwen2.5-7B)";
export const TIER3 = "Tier 3: Cloud Forensic Defender (Gemma 4)";
export const CLINICAL = "Downstream Agent: Clinical RAG Assistant (Mistral-7B)";

export const TIER_BY_SOURCE = {
  EDGE_VECTOR_CACHE: TIER1,
  EDGE_INTENT_TRIAGE: TIER2,
  CLOUD_DEFENDER_HOTPATCH: TIER3,
};
