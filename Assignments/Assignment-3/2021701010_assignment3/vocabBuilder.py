from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab, build_vocab_from_iterator
import io
import argparse

def yield_tokens(filepath, tokenizer):
    with io.open(filepath, encoding="utf8") as f:
        for text in f:
            yield tokenizer(text)


def build_vocab(filepath, tokenizer):
    v = build_vocab_from_iterator(yield_tokens(filepath, tokenizer), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    v.set_default_index(v["<unk>"])
    return v

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    # Parse the argument
    args = parser.parse_args()
    # Construct tokenizer
    if args.lang == "en":
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    elif args.lang == "fr":
        tokenizer = get_tokenizer('spacy', language="fr_core_news_sm")

    # Build Vocabulary...
    vocab = build_vocab(args.filepath, tokenizer)