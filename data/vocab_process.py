import os


def vocab_process(data_dir):
    intent_vocab_text = 'intent_label.txt'
    slot_vocab_text = 'slot_label.txt'
    # define data_dir
    train_dir = os.path.join(data_dir, 'train')
    # build intent vocab label

    with open(os.path.join(train_dir, 'label'), 'r', encoding='utf-8') as f_r, open(os.path.join(data_dir, intent_vocab_text),
                                                                                    'w', encoding='utf-8') as f_w:
        intent_vocab = set()
        for line in f_r:
            line = line.strip()
            intent_vocab.add(line)

        addition_token = ['UNK']
        for t in addition_token:
            f_w.write(t+'\n')
        intent_vocab = sorted(list(intent_vocab))
        for w in intent_vocab:
            f_w.write(w+'\n')

    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f_r, open(os.path.join(data_dir, slot_vocab_text),
                                                                                      'w', encoding='utf-8') as f_w:
        # build slot label vocab
        slot_vocab = set()
        for line in f_r:
            line = line.strip()
            slots = line.split()
            for s in slots:
                slot_vocab.add(s)
        # sorted slot vocab
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))

        # save slot_vocab
        addition_token = ['PAD', 'UNK']
        for t in addition_token:
            f_w.write(t+'\n')
        for s in slot_vocab:
            f_w.write(s+'\n')


if __name__ == '__main__':
    # build vocab
    vocab_process('atis')
    vocab_process('snips')
