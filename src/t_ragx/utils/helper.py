def get_preceding_text(text_list, max_sent=3):
    out_list = []
    for i in range(len(text_list)):
        out_list.append(text_list[max(i - max_sent, 0):i])

    return out_list
