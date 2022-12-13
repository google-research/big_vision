/**
 * @license
 * Copyright Big Vision Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {stringToChars, TOKEN_SEPARATOR, Vocabulary, Tokenizer as TokenizerInterface} from './common';

interface Candidate {
  piece: string;
  pos: number;
  score: number;
}

const scoreDesc = (a: Candidate, b: Candidate) => b.score - a.score;

function processInput(str: string): string {
  const normalized = str.normalize('NFKC');
  return normalized.length > 0 ?
    TOKEN_SEPARATOR + normalized.replace(/ /g, TOKEN_SEPARATOR) :
    normalized;
}

/**
 * Sentencepiece tokenizer implementing the BPE algorithm.
 */
export class Tokenizer implements TokenizerInterface {

  // piece -> [score, index]
  private readonly map: Map<string, [number, number]>;

  constructor(vocabulary: Vocabulary) {
    this.map = new Map<string, [number, number]>();
    vocabulary.forEach(([piece, score], idx) => {
      if (this.map.has(piece)) {
        throw new Error(`Piece "${piece}" occurs multiple times in vocabulary`);
      }
      this.map.set(piece, [score, idx]);
    });
  }

  encode(input: string): number[] {
    const processed: string = processInput(input);
    let pieces: string[] = stringToChars(processed);

    while (true) {
      const candidates: Candidate[] = [];
      for (let i = 0; i < pieces.length - 1; i++) {
        const fused = pieces[i] + pieces[i + 1];
        const el = this.map.get(fused);
        if (el) {
          candidates.push({ piece: fused, pos: i, score: el[0] });
        }
      }
      if (candidates.length === 0) {
        break;
      }
      candidates.sort(scoreDesc);
      const best = candidates[0];
      pieces = [
        ...pieces.slice(0, best.pos),
        best.piece,
        ...pieces.slice(best.pos + 2)
      ];
    }

    return pieces.map(piece => this.map.get(piece)![1]);
  }
}
