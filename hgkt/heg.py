import numpy as np


def build_exercise_graph(train_seqs, question_start, question_num):
    """Build direct support graph over question nodes using transition counts."""
    counts = np.zeros((question_num, question_num), dtype=np.float32)
    for seq in train_seqs:
        for i in range(len(seq) - 1):
            q_u = int(seq[i][1]) - question_start
            q_v = int(seq[i + 1][1]) - question_start
            if 0 <= q_u < question_num and 0 <= q_v < question_num and q_u != q_v:
                counts[q_u, q_v] += 1.0

    # Symmetrize and binarize to keep sparse support structure.
    counts = counts + counts.T
    adj = (counts > 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def build_assignment_from_skill_matrix(skill_matrix, question_start):
    """
    Build exercise->schema assignment from skill matrix.
    We use the dominant skill as schema proxy when text clustering is unavailable.
    """
    skill_num = skill_matrix.shape[0]
    question_part = skill_matrix[:, question_start:]
    question_num = question_part.shape[1]
    assignment = np.zeros((question_num, skill_num), dtype=np.float32)
    for q in range(question_num):
        owners = np.flatnonzero(question_part[:, q] > 0)
        if owners.size > 0:
            assignment[q, owners[0]] = 1.0
        else:
            # Fallback: isolate as skill-0 schema to keep tensor shapes stable.
            assignment[q, 0] = 1.0
    return assignment

