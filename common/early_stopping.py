def stop_early_naive(eval_losses: list[float]):
    """
    If the the 5th last epoch is not improved by the 4 epochs following it
    then stop.
    """
    if len(eval_losses) < 5:
        return False

    base_line = eval_losses[-5]

    for i in range(-4, 0):
        if eval_losses[i] <= base_line:
            return False

    return True
