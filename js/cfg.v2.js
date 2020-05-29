const fs = require(`fs`);
const join = require('path').join;

class Config {
	constructor() {
		// Paths
		this.audio_path = join("audio");

		// Options
		this.use_checkpoints = true; // (bool)
		this.use_checkpoints_model = true; // (bool)

		// Files
		this.audio_startpoint = 10; // (int)

		// Model
		this.pre_emphasis = 0.97; // 0.97 0.95 | (%)
		this.nfft = 2048; // 512 | lenght of single cepstral
		this.nfilt = 26; // 26 | number of filters (summary of a small frequency range) per cepstral
		this.nstep = 512;

		this.fsize = 512;
		this.fstep = 448;

		this.num_ceps = 13; // 13 | number of amount of cepstral top to down
		this.frame_rate = 16000; // data per second
		this.cep_lifter = 1;

		this.modes = ['conv', 'time', 'new'];
		this.mode = 0; // 0: conv; 1: time
		this.capsuled = true;

		//Files
		this.samples_per_file = 16; // amount of parts to look at (per file)
		this.sample_length = 0.5; // length of single sample part in second
		this.files_per_class = false; // self explaining
		this.val_files_per_class = 64; // self explaining

		// Edit
		this.step = Math.pow(2, Math.round(Math.log(this.frame_rate * this.sample_length) / Math.log(2))); // get amout of data for given sample length
		this.nstep = getSamplesFromMillis(this.nstep, this.frame_rate);
		this.nsize = getSamplesFromMillis(this.nsize, this.frame_rate);
		// number of amount of cepstral
		this.num_frames = Math.round(this.step / this.nstep);

		// Training
		this.learning_rate = 0.0001;
		this.shuffle = true // (bool)
		this.shuffle_fit = true // (bool)
		this.epochs = 5; // (int)
		this.calls = 20;
		this.batch_size = 128 * 2; // (int)


		// Dependencies
		this.categories = fs.readdirSync(this.audio_path).length;
		this.id = getIDv2(this);
		this.files = getFiles(this);

		this.files_per_class = getFilesPerClass(this);

		this.files_len = this.categories * this.files_per_class * this.samples_per_file;
		this.files_len = Math.round(this.files_len);
		this.validation_len = Math.min(this.categories * this.val_files_per_class, this.files_len);

		this.model_path = `file://./models/${this.id}/`;
		this.data_path = join(`data`, this.id + `.json`);
		this.history_path = join(`data`, this.id + `-history.json`);
		this.chart_acc_path = join("images", "acc", this.id + `.png`);
		this.chart_loss_path = join("images", "loss", this.id + `.png`);
	}
}


function getIDv2(config) {
	let id = 1;

	let ignored = ["use_checkpoints", "use_checkpoints_model", "modes", "path", "learning_rate", "shuffle", "shuffle_fit", "epochs", "batch_size", "calls", "val_files_per_class", "history_path"];
	Object.keys(config).forEach(key => {
		if (ignored.includes(key)) return;
		let value = config[key];
		let key_length = key.toString().length;
		if (isNaN(value)) value = Math.round(key_length * Math.PI * 100) / 100;
		let value_length = value.toString().length;
		id += (value * value * 100) / (value_length * key_length);
	});

	id = `${config.modes[config.mode]}-${Math.round(id*100)}${config.categories}`;
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

module.exports = {
	Config: Config,
};