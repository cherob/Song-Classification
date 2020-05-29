const trn = require('./trainer');
const cfg = require('./cfg.v2');
const dt = require('./dataset');
const util = require('./util.js');

console.log('=== Load Config File ===');
let config = new cfg.Config();
util.calculate(config);

console.log('=== Build/Load samples (press any key) ===');
let dataset = new dt.Dataset(config);
if (config.use_checkpoints) dataset.load()
dataset.build()


if (config.files_len >= 100) {
    console.log('=== Loading Model ===');
    let trainer = new trn.Trainer(config);
    if (config.mode == 0)
        trainer.generateModelConv(dataset.shape);
    else if (config.mode == 1)
        trainer.generateModelReccurent(dataset.shape);
    else if (config.mode == 2)
        trainer.generateModelNewVersion(dataset.shape);

    console.log('=== Train model ===');
    trainer.start(dataset);
}