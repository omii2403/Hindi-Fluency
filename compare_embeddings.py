import argparse
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_text(val: str) -> str:
	# Matches Ankit notebook: NFKC normalize + strip + lowercase.
	return unicodedata.normalize("NFKC", str(val).strip().lower())


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
	denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
	if denom == 0:
		return 0.0
	return float(np.dot(vec_a, vec_b) / denom)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compare multilingual embeddings exactly like the notebook flow "
			"(normalize_text + SentenceTransformer encode with batch_size=64)."
		)
	)
	parser.add_argument("word_a", nargs="?", default="शेर", help="First word")
	parser.add_argument("word_b", nargs="?", default="sher", help="Second word")
	args = parser.parse_args()

	print("Loading multilingual transformer model...")
	embed_model = SentenceTransformer("google/muril-base-cased")
	print(f"Model loaded - embedding dimension: {embed_model.get_sentence_embedding_dimension()}")

	norm_a = normalize_text(args.word_a)
	norm_b = normalize_text(args.word_b)
	norm_words = [norm_a, norm_b]

	# Matches notebook call signature.
	sem_vecs = embed_model.encode(norm_words, show_progress_bar=False, batch_size=64)
	vec_a = np.array(sem_vecs[0], dtype=float)
	vec_b = np.array(sem_vecs[1], dtype=float)

	diff = vec_a - vec_b

	print("\n=== Input ===")
	print(f"word_a(raw): {args.word_a}")
	print(f"word_b(raw): {args.word_b}")
	print(f"word_a(normalized): {norm_a}")
	print(f"word_b(normalized): {norm_b}")

	print("\n=== Embedding Difference Metrics ===")
	print(f"Cosine similarity: {cosine_similarity(vec_a, vec_b):.6f}")
	print(f"L2 distance: {np.linalg.norm(diff):.6f}")
	print(f"L1 distance: {np.sum(np.abs(diff)):.6f}")
	print(f"Max absolute dimension gap: {np.max(np.abs(diff)):.6f}")

	print("\n=== First 10 Dimensions (for sanity check) ===")
	print(f"vec_a[:10]: {np.array2string(vec_a[:10], precision=6, separator=', ')}")
	print(f"vec_b[:10]: {np.array2string(vec_b[:10], precision=6, separator=', ')}")
	print(f"diff[:10]:  {np.array2string(diff[:10], precision=6, separator=', ')}")


if __name__ == "__main__":
	main()
