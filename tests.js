var corpus = [
    "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain.",
    " These cases are perfectly simple and easy to distinguish.",
    " In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided.",
    " But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted.",
    " The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."
];


test("tokenizers", function() {
    var tok = nlpjs.tokenizers.nonAlphanumeric,
        s1 = "Hello, World!\nWill this tokenizer work? Ümlauts do not count as alphanumeric.",
        a1 = ["hello", "world", "will", "this", "tokenizer", "work", 
            "mlauts", "do", "not", "count", "as", "alphanumeric"],
        result, 
        i;

    result = tok(s1);
    deepEqual(a1, result, "nonAlphanumeric");
});


test("TermOccurrenceModel", function() {
    var to = new nlpjs.statistics.TermOccurrenceModel(),
        all,
        individual = [],
        i,
        tokens = nlpjs.tokenizers.nonAlphanumeric(corpus[4]),
        allChecked = [1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 2, 2, 2, 1, 5, 1, 2,
            5, 1, 3, 1, 3, 5, 1, 2, 1, 2, 1, 2, 1, 1, 1, 5, 2, 5, 1, 2, 2, 1,
            2, 1, 5, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 5, 1, 1,
            1, 2, 1, 1, 1, 5, 2],
        allFromSingle = {};

    for (i=0; i<tokens.length; i++) {
        individual.push(to.score(tokens[i], tokens));
        allFromSingle[tokens[i]] = individual[i];
    }
    deepEqual(individual, allChecked, "score");
    
    all = to.scoreAll(tokens);
    deepEqual(all, allFromSingle, "scoreAll");
});


test("TermFrequencyModel", function() {
    var tokens = nlpjs.tokenizers.nonAlphanumeric(corpus[4]),
        tf = new nlpjs.statistics.TermFrequencyModel(),
        allFromSingle = {},
        individual = [],
        expected = [0.014285714285714285, 0.07142857142857142,
            0.014285714285714285, 0.014285714285714285, 0.014285714285714285,
            0.014285714285714285, 0.014285714285714285, 0.014285714285714285,
            0.014285714285714285, 0.07142857142857142, 0.014285714285714285,
            0.014285714285714285, 0.02857142857142857, 0.02857142857142857,
            0.02857142857142857, 0.014285714285714285, 0.07142857142857142,
            0.014285714285714285, 0.02857142857142857, 0.07142857142857142,
            0.014285714285714285, 0.04285714285714286, 0.014285714285714285,
            0.04285714285714286, 0.07142857142857142, 0.014285714285714285,
            0.02857142857142857, 0.014285714285714285, 0.02857142857142857,
            0.014285714285714285, 0.02857142857142857, 0.014285714285714285,
            0.014285714285714285, 0.014285714285714285, 0.07142857142857142,
            0.02857142857142857, 0.07142857142857142, 0.014285714285714285,
            0.02857142857142857, 0.02857142857142857, 0.014285714285714285,
            0.02857142857142857, 0.014285714285714285, 0.07142857142857142,
            0.014285714285714285, 0.014285714285714285, 0.014285714285714285,
            0.02857142857142857, 0.014285714285714285, 0.02857142857142857,
            0.014285714285714285, 0.014285714285714285, 0.014285714285714285,
            0.014285714285714285, 0.02857142857142857, 0.014285714285714285,
            0.04285714285714286, 0.014285714285714285, 0.014285714285714285,
            0.014285714285714285, 0.07142857142857142, 0.014285714285714285,
            0.014285714285714285, 0.014285714285714285, 0.02857142857142857,
            0.014285714285714285, 0.014285714285714285, 0.014285714285714285,
            0.07142857142857142, 0.02857142857142857],
        i;

    for (i=0; i<tokens.length; i++) {
        individual.push(tf.score(tokens[i], tokens));
        allFromSingle[tokens[i]] = individual[i];
    }
    // the float results might be different on other cpus. 
    // TODO implement a deepEquals tolerant to rounding errors
    deepEqual(expected, individual, "score");
    deepEqual(allFromSingle, tf.scoreAll(tokens), "scoreAll");
});

test("TfidfModel", function() {
    var l = [],
        i,
        t; 
    for (i=0; i<corpus.length; i++) {
        l.push(nlpjs.tokenizers.nonAlphanumeric(corpus[i]));
    }
    t = new nlpjs.statistics.TfidfModel(l);
    console.log(t);

    // TODO get scores from other software to compare

    for (i=0; i<l.length; i++) {
        //console.log(corpus[i]);
        //console.log(t.scoreAll(l[i]));
    }
    ok(true, "it compiles ;-)");
});


test("VectorDocumentModel", function() {
    var corp = [], 
        tfidfModel, vectorModel, i,
        v1, v2, v3;

    for (i=0; i<corpus.length; i++) {
        corp[i] = nlpjs.tokenizers.nonAlphanumeric(corpus[i]);
    }

    tfidfModel = new nlpjs.statistics.TfidfModel(corp);
    vectorModel = new nlpjs.VectorDocumentModel(corp, tfidfModel);
    console.log(vectorModel);
    v1 = vectorModel.asVector(corp[4]);
    v2 = vectorModel.asVector(corp[5]);
    v3 = vectorModel.asVector(corp[1]);
    console.log(corpus[4]);
    console.log(v1);
    console.log(corpus[5]);
    console.log(v2);
    console.log(corpus[1]);
    console.log(v3);
    console.log("v1-v2: " + vectorModel.cosineSimilarity(v1, v2));
    console.log("v1-v3: " + vectorModel.cosineSimilarity(v1, v3));
    console.log("v1-v1: " + vectorModel.cosineSimilarity(v1, v1));

    ok(true, "it compiles...");
});
