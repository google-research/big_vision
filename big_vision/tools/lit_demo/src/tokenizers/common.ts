/**
 * @fileoverview Utility code shared between tokenizers.
 */

/**
 * A vocabulary consists of a list of tokens, and optional numerical value.
 * The numerical value is used by the unigram algorithnm to find the best
 * tokenizaion, and is ignored by the BPE algorithm.
 */
export type Vocabulary = Array<[string, number]>;

/**
 * Converts a string to a sequence of tokens.
 */
export interface Tokenizer {
  encode(input: string): number[];
}

/**
 * Factory for new `Tokenizer`.
 */
export interface TokenizerConstructor {
  new (vocabulary: Vocabulary): Tokenizer;
}

/**
 * Unicode-aware character iteration of strings.
 */
export const stringToChars = (input: string): string[] => {
  const symbols = [];
  for (const symbol of input) {
    symbols.push(symbol);
  }
  return symbols;
};

/**
 * Special separator character used to delimit sub-word tokens.
 */
export const TOKEN_SEPARATOR =
  '\u2581';  // This is the unicode character 'lower one eighth block'.
