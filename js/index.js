const trn = require('./trainer');
const cfg = require('./cfg');
const dt = require('./dataset');

console.log('=== Load Config File ===');
let config = new cfg.Config();
config.calculate();

console.log('=== Build/Load samples (press any key) ===');
let dataset = new dt.Dataset(config);
if (!config.use_checkpoints || !dataset.load())
    dataset.build();

console.log('=== Loading Model ===');
let trainer = new trn.Trainer(config);
trainer.generateModel(dataset.shape);

console.log('=== Train model ===');
trainer.start(dataset);