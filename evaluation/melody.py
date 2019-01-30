import mir_eval

def voicing_accuracy(ref_voicing, est_voicing):
    mir_eval.melody.validate_voicing(ref_voicing, est_voicing)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.

    # Count True Positives
    TP = (ref_voicing*est_voicing).sum()
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()

    return (TP + TN) / float(ref_voicing.shape[0])


def overall_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.

    raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent)
    n_voiced = ref_voicing.sum()
    TP = raw_chroma * float(n_voiced)
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()
    return (TP + TN) / float(ref_cent.shape[0])
