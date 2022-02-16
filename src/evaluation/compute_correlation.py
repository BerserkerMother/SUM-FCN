from scipy import stats


def evaluate_scores(predicted_summary, user_scores):
    kendal, spearman = [], []
    for i in range(user_scores.shape[0]):
        user_score = user_scores[i]
        # Compute kendal and spearman
        pS = stats.spearmanr(stats.rankdata(-predicted_summary),
                             stats.rankdata(-user_score))[0]
        spearman.append(pS)
        kT = stats.kendalltau(stats.rankdata(-predicted_summary),
                              stats.rankdata(-user_score))[0]
        kendal.append(kT)
    return sum(kendal) / len(kendal), sum(spearman) / len(spearman)
