const cfg = require('./cfg.v2');
const dt = require('./dataset');
const util = require('./util.js');
const fs = require(`fs`);
const wf = require('wavefile');
const cliProgress = require('cli-progress');
const MFCC = require('./mfcc/mfcc.js');
const numpy = require('numjs');
const join = require('path').join;
const Config = require('./cfg.v2').Config;

let qualityRange = 10;

const tf = require("@tensorflow/tfjs-node-gpu");
// require('@tensorflow/tfjs-node');

console.log('=== Load Config File ===');
let config = new cfg.Config();
util.calculate(config);

let mfcc_conf = {
    cep_lifter: config.cep_lifter,
    frame_rate: config.frame_rate,
    fsize: config.fsize,
    fstep: config.fstep,
    nfft: config.nfft,
    nfilt: config.nfilt,
    nstep: config.nstep,
    num_ceps: config.num_ceps,
    pre_emphasis: config.pre_emphasis
}

tf.loadLayersModel(config.model_path + '/model.json').then((model) => {
    fs.readdirSync('test').forEach(file => {
        let file_localpath = join(
            __dirname,
            '..',
            'test',
            file
        );

        let X = [];
        let results = [];
        let buffer = fs.readFileSync(file_localpath);

        let wav = new wf.WaveFile(buffer);

        if (wav.fmt.sampleRate !== config.frame_rate)
            wav.toSampleRate(config.frame_rate);

        let wavBuffer = wav.toBuffer();
        for (let index = 0; index < qualityRange; index++) {

            let random_offset = Math.floor(Math.random() * (wavBuffer.length - config.step)) + config.step
            let signal = wavBuffer.slice(random_offset, random_offset + config.step)

            // load cepstral parts
            let MFCCs = new MFCC(mfcc_conf)
            let mfccs = MFCCs.getMFCCs(signal);

            mfccs = mfccs.map(c1 => c1.map(c2 => formatValue(c2, config.capsuled)));
            X.push(mfccs);
        }

        X = tf.tensor(X);
        results = model.predict(X).arraySync();
        let resultAverage = new Array(results[0].length).fill(0);
        results.forEach(result => {
            result.forEach((value, index) => {
                resultAverage[index] += value;
            })
        });

        console.log(`${file}`);
        console.log(`-> ${Object.keys(config.files)[resultAverage.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0)]} (${Math.round(Math.max(...resultAverage)*1000)/100}%)`)
    });


});

function formatValue(value = Array, innerValue = false) {
    if (value && value[0] && value[0][0])
        return innerValue ? value[0] : value[0][0];
    else if (value[0])
        return innerValue ? value : value[0];
    else
        return innerValue ? [value] : value;
}