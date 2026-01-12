def generate_explanation(
    activation_coverage,
    fake_similarity,
    real_similarity,
    activated_frames
):
    """
    activation_coverage: float (0–1)
    fake_similarity: float (0–1)
    real_similarity: float (0–1)
    activated_frames: int
    """

    # ---- Rule 1: Strong synthetic manipulation ----
    if (
        activation_coverage > 0.25 and
        fake_similarity > 0.75 and
        activated_frames >= 5
    ):
        return (
            "Consistent activation observed across multiple frames, "
            "with localized emphasis on facial boundary regions. "
            "The highlighted regions show high similarity to known "
            "synthetic blending artifacts, indicating manipulated content."
        )

    # ---- Rule 2: Weak or isolated manipulation ----
    if (
        activation_coverage > 0.15 and
        fake_similarity > real_similarity and
        activated_frames < 5
    ):
        return (
            "Localized activation detected in a small number of frames. "
            "The visual patterns exhibit moderate similarity to synthetic "
            "artifacts, suggesting possible but limited manipulation."
        )

    # ---- Rule 3: Likely authentic ----
    if real_similarity > fake_similarity:
        return (
            "Model activations are weak and inconsistent across frames. "
            "Highlighted regions show greater similarity to authentic "
            "visual patterns, supporting an authentic classification."
        )

    # ---- Fallback rule ----
    return (
        "The system detected ambiguous visual evidence. "
        "While certain regions contributed to the decision, "
        "the confidence of manipulation remains low."
    )
