const fs = require(`fs`);

class Dataset {

    constructor(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_lenght = 512, num_segments = 5) {
        this.config = arguments;
        this.data = {
            mapping: [],
            X: [],
            y: {}
        }

        fs.readdirSync(dataset_path).forEach(dir => {
            this.data.mapping.push(dir);
        });
        console.log(this)
    }

    save() {

        fs.readdirSync(dataset_path).forEach(dir => {
            files[dir] = [];
            fs.readdirSync(join(dataset_path, dir)).forEach((file) => {
                files[dir].push(file);
            });
        });

    }

}

module.exports = {
    Dataset: Dataset,
};