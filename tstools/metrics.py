def error(y, y_fcst):
    err = y - y_fcst
    return err


def mae(y, y_fcst):
    mae_value = error(y, y_fcst).abs().mean()
    return mae_value


def mse(y, y_fcst):
    sqe = error(y, y_fcst)**2
    mse_value = sqe.mean()
    return mse_value


def wape(y, y_fcst):
    err = error(y, y_fcst)
    wape_value = 100 * err.abs().sum() / y.abs().sum()
    return wape_value
