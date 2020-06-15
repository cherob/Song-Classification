const fs = require(`fs`);
const wf = require('wavefile');
const cliProgress = require('cli-progress');
const numpy = require('numjs');
const util = require('../util/util');

const mfccs = require('../components/mfcc.js').mfccs;

class Dataset {
    constructor(config, location = 'audio') {
        this.location = location;
        this.config = config
        this.data = {
            mapping: [],
            X: [],
            y: []
        }

        fs.readdirSync(this.location).forEach(dir => {
            this.data.mapping.push(dir);
        });
    }

    shape() {
        let s = util.shape(this.data.X)
        return s.slice(1, s.length);
    };

    generate() {
        console.log(' -> generate Model!')

        let loadlist = [];
        fs.readdirSync(this.location).forEach((dir, index) => {
            fs.readdirSync([this.location, dir].join('/')).forEach((file) => {
                loadlist.push({
                    name: file,
                    path: [this.location, dir, file].join('/'),
                    genre: dir,
                    index: index
                })
            })
        });

        let bar_options = [{
            format: `Process Files [{bar}] | {percentage}% || {value}/{total} Chunks | {file}`
        }, cliProgress.Presets.shades_classic];
        let bar = new cliProgress.SingleBar(...bar_options);

        bar.start(loadlist.length, 0, {
            file: ''
        });

        loadlist.forEach((file, index) => {
            let samples = this.process(file.path);
            let labels = new Array(samples.length).fill(file.index)

            this.data.X.push(...samples);
            this.data.y.push(...labels);

            bar.update(index + 1, {
                file: file.name
            });
        })
        bar.stop();

        let min = 0;
        let max = 0;

        this.data.X.forEach(m => m.forEach(c => c.forEach(v => {
            min = Math.min(min, v);
            max = Math.max(max, v);
        })));

        this.config.data.max = max;
        this.config.data.min = min;

        this.data.X = this.data.X.map(m => m.map(c => util.compress(c, min, max)));

        this.save();
        return this.data;
    }

    process(file) {
        let buffer = fs.readFileSync(file);
        let wav = new wf.WaveFile(buffer);
        if (wav.fmt.sampleRate !== this.config.data.sample_rate) {
            wav.toSampleRate(this.config.data.sample_rate);
            fs.writeFileSync(file, wav.toBuffer());
        }

        let wav_buffer = wav.toBuffer();
        wav_buffer = wav_buffer.filter((_, i) => (i + 1) % 2 === 0)

        wav_buffer = wav_buffer.slice(wav_buffer.length * (this.config.data.audio_cut / 2),
            wav_buffer.length * (1 - (this.config.data.audio_cut / 2)));

        let samples = new Array(this.config.data.samples_per_file).fill([]);
        samples = samples.map((sample, index) => {
            let offset = wav_buffer.length / (this.config.data.samples_per_file)
            let start_point = offset * index;
            let end_point = start_point + (this.config.data.sample_rate * this.config.data.sample_lenght);

            sample = wav_buffer.slice(start_point, end_point);
            let coefs = mfccs(sample, this.config.data);

            coefs = coefs.map(mfcc => mfcc.map(value => -value));
            // coefs = util.transpose(coefs);

            return coefs;
        });

        return samples;
    }

    save() {
        console.log(` -> save test/train data...`)
        fs.writeFileSync(this.config.path.database, JSON.stringify(this.data));
        fs.writeFileSync(this.config.path.config, JSON.stringify(this.config));
    }

    load() {
        try {
            let stringData = fs.readFileSync(this.config.path.database);
            let jsonData = JSON.parse(stringData);
            if (jsonData.X.length)
                this.data = jsonData;
            console.log(` -> loaded data (${this.data.X.length} samples)`);
            return true;
        } catch (error) {
            this.generate();
        }
    }
}

module.exports = {
    Dataset: Dataset,
};