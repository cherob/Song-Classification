const fs = require(`fs`);
const join = require('path').join;

function getID(config) {
	let id = 1;

	let keys = ["nfft", "nfilt", "nfeat",
		"lowfreq", "highfreq", "frame_rate",
		"samples_per_file", "sample_length",
		"files_per_class", "nstep", "nsize"
	];

	keys.forEach(key => {
		let value = config[key];
		if (!value) value = Math.round(Math.PI * 100) / 100;
		let key_length = key.toString().length;
		let value_length = value.toString().length;


		id += (value * value) / (value_length * key_length);
	});
	id = `${Math.round(id)}${config.categories}`;
	return id;
}

function getFiles(config) {
	let files = {};

	fs.readdirSync(config.audio_path).forEach(dir => {
		files[dir] = [];
		fs.readdirSync(join(config.audio_path, dir)).forEach((file, index) => {
			if (!config.files_per_class || index < config.files_per_class)
				files[dir].push(file);
		});
	});
	return files;
}

function getFilesPerClass(config) {
	let files_per_class = 0;
	Object.keys(config.files).forEach(dir => {
		if (files_per_class) {
			if (files_per_class > config.files[dir].length) files_per_class = config.files[dir].length;
		} else {
			files_per_class = config.files[dir].length;
		}
	});
	return files_per_class;
}

function getSamplesFromMillis(ms, rate) {
	let frame_per_milli = rate / 1000
	let samples = frame_per_milli * ms;
	return samples;
}

class Config {
	constructor() {
		// Paths
		this.audio_path = join("audio");

		// Options
		this.use_checkpoints = true; // (bool)

		// Files
		this.audio_startpoint = 10; // (int)
		this.audio_length = 120; // (int)

		// Model
		this.pre_emphasis = 0.97 // 0.97 0.95 | (%)
		this.nsize = 25 // 25 | (ms) window size of actually cepstal
		this.nstep = 10 // 10  | (ms) step size for cepstal overlap
		this.nfft = 512; // 512 | lenght of single cepstral
		this.nfilt = 40; // 26 | number of filters (summary of a small frequency range) per cepstral
		this.nfeat = false; // 13 | number of amount of cepstral 
		this.frame_rate = 16000; // data per second

		this.modes = ['conv', 'time']
		this.mode = 1 // 0: conv; 1: time

		//Files
		this.samples_per_file = 5; // amount of parts to look at (per file)
		this.sample_length = 1.5; // length of single sample part in second
		this.files_per_class = 1; // self explaining
		this.validation_len = 1 //  validation files per categories

		// Edit
		this.step = Math.pow(2, Math.round(Math.log(this.frame_rate * this.sample_length) / Math.log(2))); // get amout of data for given sample length
		this.nstep = getSamplesFromMillis(this.nstep, this.frame_rate);
		this.nsize = getSamplesFromMillis(this.nsize, this.frame_rate);
		// number of amount of cepstral
		if (!this.nfeat)
			this.nfeat = Math.round(this.step / this.nstep);

		// Training
		this.learning_rate = 0.001;
		this.shuffle = false // (bool)
		this.shuffle_fit = true // (bool)
		this.epochs = 5; // (int)
		this.batch_size = 1024; // (int)
		this.calls = false; // (bool, int)

		// Range
		this.min = Number.MAX_VALUE;
		this.max = 0;

		// Dependencies
		this.categories = fs.readdirSync(this.audio_path).length;
		this.id = getID(this);
		this.files = getFiles(this);
		this.files_per_class = getFilesPerClass(this);
		this.files_len = this.categories * this.files_per_class * this.samples_per_file;
		this.files_len = Math.round(this.files_len);
		this.validation_len = this.validation_len * this.categories;
		if (this.validation_len > this.files_len)
			this.validation_len = this.files_len / this.samples_per_file;


		// this.model_audio_date_path = join(`data`, `model` + this.id + `.csv`);
		// this.predictions_date_path = join(`data`, `predictions` + this.id + `.csv`);
		this.model_path = `file://./models/${this.id}/`;
		this.data_path = join(`data`, this.id + `.json`);
		this.chart_acc_path = join("images", "acc", this.id + `.png`);
		this.chart_loss_path = join("images", "loss", this.id + `.png`);
	}

	calculate() {
		console.log(` ${this.categories} loaded categories`);
		console.log(`   with  ${this.files_per_class} files each,`);
		console.log(`   every file is ${this.sample_length} seconds long,`);
		console.log(`   divided into ${this.samples_per_file} samples`);
		console.log(`   with (${this.nfilt}x${this.nfeat}) (filters, ccepstral)`);
		console.log(` -> results in ${this.files_len} data records`);
		console.log(`    ID : ${this.id} MODE: ${this.modes[this.mode]}`);
	}
}

module.exports = {
	Config: Config,
};