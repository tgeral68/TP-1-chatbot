class BPETokenizer:
    def __init__(self, vocabulary_size = 100):
        self.vocab = {}
        self.ivocab = {}
        self.vocabulary_size = vocabulary_size
        self.special_tokens = {}
    
    def get_pairs_frequency(self, actual_split, words_frequency):
        ''' Compute frequency of all pairs in actual split
        '''
        pairs_frequency = {}

        for word, word_split in actual_split.items():
            
            wf = words_frequency[word]
            for lv_index in range(len(word_split) - 1):
                pair = (word_split[lv_index], word_split[lv_index + 1])
                if pair not in pairs_frequency:
                    pairs_frequency[pair] = 0
                pairs_frequency[pair] += words_frequency[word]
        return pairs_frequency

    def merge(self, pair, actual_split, words_frequency):
        ''' Merge all pairs, it is an inplace operation on actual_split

        '''
        for word in words_frequency:
            # Merging all the splits of the curent split considering the pairs
            word_split = actual_split[word]
            lv_index = 0
            while lv_index < len(word_split) - 1:
                # We merge the split if 
                if (word_split[lv_index] == pair[0] and word_split[lv_index + 1] == pair[1]) : 
                    word_split[lv_index] = pair[0] + pair[1]
                    del word_split[lv_index + 1]
                lv_index += 1
        return actual_split

    def fit(self, text_collection):
        self.merges_rules = []
        words_frequency = {}
        for text in text_collection:
            splitted_text = ["_"+t if(i!=0) else t for i, t in enumerate(text.split())]
            for word in splitted_text:
                if word not in words_frequency:
                    words_frequency[word] = 0
                words_frequency[word] += 1
        self.vocab = {v: i for i, v in enumerate({character for word in  words_frequency.keys() for character in word})}
        actual_split = {word: [c for c in word] for word in words_frequency.keys()}
        for i in range(self.vocabulary_size):
            # we compute pairs frequencies
            pairs_frequency = self.get_pairs_frequency(actual_split, words_frequency)
            
            bpair = max(pairs_frequency, key=lambda x: x[1])
            self.merges_rules.append(bpair)
            self.vocab["".join(bpair)] = len(self.vocab)
            actual_split = self.merge(bpair, actual_split, words_frequency)
        self.ivocab = {v:k for k, v in self.vocab.items()}
        return actual_split
    
    def add_special_tokens(self, list_special_tokens):
        for i in list_special_tokens:
            if(i not in self.special_tokens):
                self.special_tokens[i] = len(self.special_tokens)
                self.vocab[i] = len(self.vocab)

    def tokenize(self, text):
        tokenized_text = [ ]
        word_tokenized = ["_"+t if(i!=0) else t for i, t in enumerate(text.split())]
        splits = [[word] if(word in special_tokens)  else [char for char in word] for word in word_tokenized]
        # we apply the rules in the same order
        for rule in self.merges_rules:
            for sp_i, sp in enumerate(splits):
                i = 0
                while i < len(sp) - 1:
                    a, b = sp[i], sp[i + 1]
                    if (a, b) == rule:
                        sp[i] = a+b
                        del sp[i+1]
                    else:
                        i += 1
                splits[sp_i]  = sp 
        return  [self.vocab[i] for i in sum(splits, [])]

    def __call__(self, text):
        splits = self.tokenize(text)
        return  [self.vocab[i] for i in sum(splits, [])]


    def decode(self, input_ids):
        decoded = []
        for i in input_ids:
            decoded.append(self.ivocab[i])
        return ''.join(decoded).replace("_",' ')
           