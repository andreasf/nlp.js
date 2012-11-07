/**
 * nlp.js
 *
 * A rather unfinished library for natural language processing tasks.
 *
 * Includes
 * - a tokenizer
 * - a vector document model (occurrence, tf, tf*idf)
 * - cosine similarity calculation
 * 
 * Usage:
 * Check tests.js for examples
 *
 * The MIT License
 *
 * Copyright (c) 2012 Andreas Fleig
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Namespacing
 * From: Stoyan Stefanov: "JavaScript Patterns", O'Reilly, 2010
 */
var nlpjs = nlpjs || {}; 
nlpjs.namespace = function(ns) {
    'use strict';
    var parts = ns.split('.'),
        parent = nlpjs,
        i;
    if (parts[0] === 'nlpjs') {
        parts = parts.slice(1);
    }   
    for (i=0; i<parts.length; i++) {
        if (typeof(parent[parts[i]]) === 'undefined') {
            parent[parts[i]] = {}; 
        }
        parent = parent[parts[i]];
    }   
    return parent;
};


(function(){
    "use strict";

    var PRUNE = false,
        PRUNE_BELOW_DF = 0,
        PRUNE_ABOVE_DF = 0.6,
        undefined;

    nlpjs.VectorDocumentModel = VectorDocumentModel;
    nlpjs.termOccurrences = termOccurrences;
    nlpjs.termFrequencies = termFrequencies;
    nlpjs.documentOccurrences = documentOccurrences;
    
    /**
     * @param document Array of Strings (terms in the document)
     * @returns map of String -> Number (occurence of each term)
     */
    function termOccurrences(document) {
        var i,
            counts = {},
            term;

        for (i=0; i<document.length; i++) {
            term = document[i];
            if (counts[term] === undefined) {
                counts[term] = 1;
            } else {
                counts[term]++;
            }
        }
        return counts;
    }
    
    
    /**
     * @param document Array of Strings (terms in the document)
     * @returns map of String -> Number (frequency of each term)
     */
    function termFrequencies(document) {
        var key,
            frequencies,
            count;

        frequencies = termOccurrences(document);
        count = document.length;
        for (key in frequencies) {
            frequencies[key] = frequencies[key]/count;
        }
        return frequencies;
    }

    
    /**
     * @param corpus String[][] of documents, terms
     * @returns map of String -> Number (occurence of each term)
     */
    function documentOccurrences(corpus) {
        var i, occs, key, 
            docOccs = {};

        for (i=0; i<corpus.length; i++) {
            occs = termOccurrences(corpus[i]);
            for (key in occs) {
                if (docOccs[key] === undefined) {
                    docOccs[key] = 1;
                } else {
                    docOccs[key] += 1;
                }
            }
        }
        return docOccs;
    }


    /**
     * creates a new vector space model from the given corpus
     * @param corpus String[][] of documents, terms
     * @param statisticModel model used to create term vectors, 
     *        e.g. a TfidfModel instantiated with the same corpus
     */
    function VectorDocumentModel(corpus, statisticModel) {
        var i, j,
            wordToIndex = {},
            dos,
            df,
            docnum = corpus.length,
            index = 0;
        
        if (PRUNE) {
            dos = documentOccurrences(corpus);
        }

        for (i=0; i<corpus.length; i++) {
            for (j=0; j<corpus.length; j++) {
                if (PRUNE) {
                    df = dos[corpus[i][j]] / docnum;
                    if (df < PRUNE_BELOW_DF || df > PRUNE_ABOVE_DF) {
                        continue;
                    }
                }
                wordToIndex[corpus[i][j]] = index;
                index++;
            }
        }
        this.wordToIndex = wordToIndex;
        this.dim = index;
        this.statistic = statisticModel;
    }
    
    VectorDocumentModel.prototype = {
        /**
         * returns a vector space representation of a document
         * @param document String[] of terms
         * @returns Float32Array (term vector)
         */
        asVector: function(document) {
            var buf = new ArrayBuffer(4 * this.dim),
                doc = new Float32Array(buf),
                scores = this.statistic.scoreAll(document),
                index, key;

            for (key in scores) {
                index = this.wordToIndex[key];
                doc[index] = scores[key];
            }
            return doc;
        },

        /**
         * @param a Float32Array of term vectors
         * @param b Float32Array of term vectors
         * @returns cosine similarity of a and b (as Number)
         */
        cosineSimilarity: function(a, b) {
            var i, 
                t1 = 0,
                t2 = 0,
                t3 = 0;

            for (i=0; i<a.length; i++) {
                t1 += a[i] * b[i];
                t2 += Math.pow(a[i], 2);
                t3 += Math.pow(b[i], 2);
            }
            return t1 / (Math.sqrt(t2) * Math.sqrt(t3));
        }
    }
}());


nlpjs.namespace("tokenizers");
(function(){
    "use strict";

    nlpjs.tokenizers.nonAlphanumeric = nonAlphanumeric;

    function nonAlphanumeric(str) {
        var tokens, last;
        tokens = str.toLowerCase().split(/[^\w]+/);
        if (tokens[0] === "") {
            tokens = tokens.splice(1);
        }
        last = tokens.length - 1;
        if (tokens[last] === "") {
            tokens = tokens.splice(0, last);
        }
        return tokens;
    }
}());


nlpjs.namespace("statistics");
(function(){
    "use strict";
    var undefined,
        termOccurrences = nlpjs.termOccurrences,
        termFrequencies = nlpjs.termFrequencies;

    nlpjs.statistics.TermOccurrenceModel = TermOccurrenceModel;
    nlpjs.statistics.TermFrequencyModel = TermFrequencyModel;
    nlpjs.statistics.TfidfModel = TfidfModel;
    
    termOccurrences = nlpjs.termOccurrences;
    termFrequencies = nlpjs.termFrequencies;
 
    
    function TermOccurrenceModel(corpus) {
    }
    
    TermOccurrenceModel.prototype = {
        /**
         * @param term String to get occurrence of
         * @param document Array of Strings (terms in the document)
         * @returns number of occurences of term in document
         */
        score: function(term, document) {
            var i, 
                count = 0;

            for (i=0; i<document.length; i++) {
                if (document[i] === term) {
                    count++;
                }
            }
            return count;
        },
        scoreAll: termOccurrences
    }

    
    function TermFrequencyModel(corpus) {
    }

    TermFrequencyModel.prototype = {
        /**
         * @param term String to get frequency of
         * @param document Array of Strings (terms in the document)
         * @returns frequency of term in document
         */
        score: function(term, document) {
            var i,
                count = 0;

            for (i=0; i<document.length; i++) {
                if (document[i] === term) {
                    count++;
                }
            }
            return count/document.length;
        },
        scoreAll: termFrequencies
    }


    /**
     * Creates a new tf*idf model
     * @param corpus String[][] of documents, terms
     */
    function TfidfModel(corpus) {
        var docOccs = nlpjs.documentOccurrences(corpus),
            key;

        for (key in docOccs) {
            docOccs[key] = Math.log(corpus.length / docOccs[key]);
        }
        this.idf = docOccs;
        this.corpusSize = corpus.length;
    }

    TfidfModel.prototype = {
        /**
         * Returns the tf*idf score for all given terms
         * @param document String[] of terms
         * @returns map of String -> Number: score for each term
         */
        scoreAll: function(document) {
            var tf = termFrequencies(document),
                max = 0,
                key,
                idf;

            // get maximum to normalize tf
            for (key in tf) {
                if (tf[key] > max) {
                    max = tf[key];
                }
            }
            for (key in tf) {
                idf = this.idf[key];
                if (idf === undefined) {
                    idf = Math.log(this.corpusSize/1);
                }
                tf[key] = (tf[key]/max) * idf;
            }
            return tf;
        }
    }
}());

