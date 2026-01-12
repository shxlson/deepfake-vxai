import torch


def topk_mean(frame_scores, topk_ratio=0.3):
    """
    frame_scores: 1D torch tensor of frame-level probabilities
    topk_ratio: fraction of highest-scoring frames to use
    """
    assert frame_scores.ndim == 1, "frame_scores must be 1D"

    k = max(1, int(len(frame_scores) * topk_ratio))
    topk_vals, _ = torch.topk(frame_scores, k)

    return topk_vals.mean()


if __name__ == "__main__":
    # 🔬 quick sanity test
    scores = torch.tensor([0.12, 0.91, 0.05, 0.87, 0.20, 0.76])
    video_score = topk_mean(scores)

    print("Frame scores:", scores.tolist())
    print("Video-level score:", video_score.item())
