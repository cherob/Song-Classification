const tf = require("@tensorflow/tfjs-node-gpu");
// require('@tensorflow/tfjs-node');
const stats = require('../util/analytics');
const dt = require('./Dataset');
const fs = require(`fs`);
const cliProgress = require('cli-progress');
const util = require('../util/util.js');
const structuredModel = require('../components/model');

class Trainer {
  /**
   * 
   * @param {dt.Dataset} dataset 
   */
  constructor(dataset) {
    this.dataset = dataset;

    this.config = this.dataset.config.train;
    if (!this.config.calls) this.config.calls = Number.MAX_VALUE;

    if (this.config.included)
      this.dataset.data.X = this.dataset.data.X.map(_ => _.map(x => x.map(y => [y])));

    this.model = structuredModel.reccurent(this.dataset.shape(), this.dataset.data.mapping.length);
    this.model.summary();
    console.log('=== Save Images ===');
    stats.saveImages(this.dataset.config, this.dataset.data);

    this.result_params = {
      total_batch_size: 0,
      epoch: 0,
      logs: [],
      val_logs: [],
      history: function () {
        let acc = this.logs.map((log) => log.acc);
        let loss = this.logs.map((log) => log.loss);
        let val_loss = this.val_logs.map((log) => log.val_loss);
        let val_acc = this.val_logs.map((log) => log.val_acc);
        return {
          epoch: this.epoch,
          acc: this.averageProzess(acc, 10, 2, 100),
          loss: this.averageProzess(loss, 10, 2, 10),
          val_acc: this.averageProzess(val_loss, 1),
          val_loss: this.averageProzess(val_acc, 1),
        }
      },
      /**
       * 
       * @param {Array} arr 
       * @param {Number} count 
       * @param {Number} period 
       */
      averageProzess: function (arr, count, period = 2, multi = 1) {
        if (count > arr.length)
          count = arr.length;
        if (!arr.length)
          return 0;

        arr = arr.splice(arr.length - count, count);

        let a = util.average(arr) * multi;

        a = parseFloat(a).toFixed(period);
        return a;
      },
      progress: function (size = 0) {
        this.total_batch_size += size;
        return this.total_batch_size;
      }
    };

  }

  compile() {
    let optimizer = tf.train.adam(this.config.learning_rate);

    this.model.compile({
      loss: tf.metrics.categoricalCrossentropy,
      optimizer: optimizer,
      metrics: ["accuracy"]
    });
  }

  async start() {
    let result_params = this.result_params;

    for (let i = 1; i < this.config.calls + 1; i++) {
      let X, y;
      if (this.dataset.config.check)
        try {
          this.model = await tf.loadLayersModel(`file://./${this.dataset.config.path.model}model.json`);
          let json_params = JSON.parse(
            fs.readFileSync(this.dataset.config.path.history));
          result_params.epoch = json_params.epoch;
          result_params.logs = json_params.logs;
        } catch (error) {
          console.warn(` -> couldn't load model: ${error.message}`)
        }


      this.compile();

      // Train model
      X = this.dataset.data.X
      y = util.toCategorical(this.dataset.data.y, this.dataset.data.mapping.length)

      // Loading Bar
      let status_epoch_bar_options = [{
        format: `Epoch {epoch} [{bar}] | Acc: {acc} % | Loss: {loss} % | Acc: {val_acc} % | Loss: {val_loss} % `
      }, cliProgress.Presets.shades_classic];
      let epoch_bar = new cliProgress.SingleBar(...status_epoch_bar_options);

      let validation_split = this.config.validation_split;
      let m = await this.model.fit(tf.tensor(X), tf.tensor(y), {
        verbose: 0,
        epochs: 1 | this.config.epochs,
        shuffle: true,
        batchSize: this.config.batch_size,
        validationSplit: this.config.validation_split,
        callbacks: {
          onEpochBegin() {
            result_params.epoch++;
            epoch_bar.start(X.length * (1 - validation_split), 0, result_params.history());
            process.stdout.cursorTo(0); // move cursor to beginning of line
            process.stdout.clearLine(); // clear current text
          },
          onEpochEnd() {},
          onBatchEnd(_, logs) {
            result_params.logs.push(logs);
            epoch_bar.update(result_params.progress(logs.size), result_params.history())
          }
        }
      });

      result_params.val_logs.push({
        val_acc: m.history.val_acc,
        val_loss: m.history.val_loss
      });

      epoch_bar.update(result_params.progress(), result_params.history())
      result_params.total_batch_size = 0;
      epoch_bar.stop();


      await this.model.save(`file://./${this.dataset.config.path.model}`);
      fs.writeFileSync(this.dataset.config.path.history, JSON.stringify({
        logs: result_params.logs,
        epoch: result_params.epoch
      }));
    }
  }
}

module.exports = {
  Trainer: Trainer
};