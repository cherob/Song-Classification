const fs = require(`fs`);
const path = require("path")
const removeFilePart = dirname => path.parse(dirname).dir

class Config {
    constructor(path) {
        this.path = path;
    }

    load() {
        let raw = fs.readFileSync(this.path);
        this.config = JSON.parse(raw);
        this.config.version = this.config.version.toFixed(2);

        Object.keys(this.config.path).forEach(key => {
            this.config.path[key] = this.config.path[key].replace(/%/gm, this.config.version)
            let folder = removeFilePart(this.config.path[key]);
            if (!fs.existsSync(folder)) {
                fs.mkdirSync(folder);
            }
        })

        return this.config;
    }
}

module.exports = {
    Config: Config
}