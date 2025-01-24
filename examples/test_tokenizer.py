from transformix.tokenization import PmlmTokenizer
from tokenizers import models

def main():
    
    vocab_path = "assets/vocabs/vocab-38k.txt"
    tokenizer = PmlmTokenizer(vocab_file=vocab_path)

    print(tokenizer)
    inputs = tokenizer("A VF GDE AA LLAA LLA LL")
    input_ids = inputs['input_ids']

    print(input_ids)
    print(tokenizer.convert_ids_to_tokens(input_ids))
    print(inputs['attention_mask'])

if __name__ == '__main__':
    main()