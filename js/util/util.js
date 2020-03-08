const nj = require('numjs');

// Utilities
map = (x, i1, i2, o1, o2) => ((x - i1) * (o2 - o1)) / (i2 - i1) + o1;

average = arr => arr.reduce((a,b) => a + b, 0) / arr.length

delay = function (ms) {
  var x = 0;
  setTimeout(function () {
    if (x) {
      return undefined;
    }
    x++;
  }, ms);
};

let msg_list = [];

function printOnce(msg) {
  if (!msg_list.includes(msg)) {
    console.log(msg);
    msg_list.push(msg);
  }
}

function shuffle(obj1, obj2) {
  var index = obj1.length;
  var rnd, tmp1, tmp2;

  while (index) {
    rnd = Math.floor(Math.random() * index);
    index -= 1;
    tmp1 = obj1[index];
    tmp2 = obj2[index];
    obj1[index] = obj1[rnd];
    obj2[index] = obj2[rnd];
    obj1[rnd] = tmp1;
    obj2[rnd] = tmp2;
  }
  return [
    obj1,
    obj2
  ];
}
flatten = (arr) => {
  return arr.reduce(function (flat, toFlatten) {
    return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
  }, []);
}

/**
 * 
 * @param {Array} y 
 * @param {Int} num_classes 
 */
function to_categorical(y, num_classes = null) {
  categorical = y.map(y_ => {
    let blank = new Array(num_classes).fill(0);
    blank[y_] = 1;
    return blank;
  });
  return categorical
}


function standardDeviation(values){
  var avg = average(values);
  
  var squareDiffs = values.map(function(value){
    var diff = value - avg;
    var sqrDiff = diff * diff;
    return sqrDiff;
  });
  
  var avgSquareDiff = average(squareDiffs);

  var stdDev = Math.sqrt(avgSquareDiff);
  return stdDev;
}

module.exports = {
  map: map,
  delay: delay,
  shuffle: shuffle,
  printOnce: printOnce,
  standardDeviation: standardDeviation,
  arrAvg: average,
  flatten: flatten,
  to_categorical: to_categorical
};