

def join_history(history1, history2):

    all_keys=set(list(history1.history.keys())+list(history2.history.keys()))

    for k in all_keys:
        if k in history2.history.keys():
            if k in history1.history.keys():
                history1.history[k].extend(history2.history[k])
            else:
                history1.history[k]=history2.history[k]

    return history1