

def single_model_val_csv(le, fold, m_type):
    if len(le) > 0:
        name = f'val_le_{fold}_single_model_{m_type}submission.csv'
    else:
        name = f'val_{fold}_single_model_{m_type}submission.csv'
    return name


def single_model_test_csv(le, fold, m_type):
    if le == '':
        name = f'test_{fold}_single_model_{m_type}submission.csv'
    else:
        name = f'test_{le}_{fold}_single_model_{m_type}submission.csv'
    return name