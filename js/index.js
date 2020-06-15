const dt = require('./core/Dataset');
const trn = require('./core/Trainer');
const cfg = require('./core/Config');

let config = new cfg.Config('./js/config.json').load();

let dataset = new dt.Dataset(config);
dataset.load();

let trainer = new trn.Trainer(dataset);
trainer.start();