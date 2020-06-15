// Utilities
combine2D = (array, otherArray) => array.forEach((value) => {
  return otherArray(value);
});

flatten = (arr) => {
  return arr.reduce(function (flat, toFlatten) {
    return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
  }, []);
};

var msg_list = []
module.exports = {
  flatten: flatten,
  transpose: (A) => A[0].map((col, i) => A.map(row => row[i])),
  map: (x, i1, i2, o1, o2) => ((x - i1) * (o2 - o1)) / (i2 - i1) + o1,
  compress: function (A, min, max) {
    A = A.map(ar => this.map(ar, min, max, 0, 1));
    return A;
  },
  average: arr => arr.reduce((a, b) => a + b, 0) / arr.length,
  map: (x, i1, i2, o1, o2) => ((x - i1) * (o2 - o1)) / (i2 - i1) + o1,
  combine: (arrays) => {
    let combinedArray = new Array(arrays.length).fill([]);
    combinedArray.forEach((part, index) => {
      arrays.forEach((array) => {
        part.push(array[index]);
      })
    });
    return combinedArray;
  },
  shuffle: function (obj1, obj2) {
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
  },
  printOnce: function (msg, id) {

    if (id) {
      if (!msg_list.includes(id)) {
        msg_list.push(id);
        console.log(msg);
      }
    } else {
      if (!msg_list.includes(msg)) {
        console.log(msg);
        msg_list.push(msg);
      }
    }
  },
  standardDeviation: function (values) {
    var avg = this.average(values);

    var squareDiffs = values.map(function (value) {
      var diff = value - avg;
      var sqrDiff = diff * diff;
      return sqrDiff;
    });

    var avgSquareDiff = this.average(squareDiffs);

    var stdDev = Math.sqrt(avgSquareDiff);
    return stdDev;
  },
  toCategorical: function (y, num_classes = null) {
    categorical = y.map(y_ => {
      let blank = new Array(num_classes).fill(0);
      blank[y_] = 1;
      return blank;
    });
    return categorical
  },
  fromCategorical: function (y, num_classes = null) {
    let category = null;
    y.forEach((y_, index) => {
      if (y_ == 1)
        category = index
    });
    return categorical
  },
  shape: (A) => ([A.length || 0,
    A ? (A[0] ? A[0].length : 0) : 0,
    A ? (A[0] ? (A[0][0] ? A[0][0].length : 0) : 0) : 0,
    A ? (A[0] ? (A[0][0] ? (A[0][0][0] ? A[0][0][0].length : 0) : 0) : 0) : 0,
    A ? (A[0] ? (A[0][0] ? (A[0][0][0] ? (A[0][0][0][0] ? A[0][0][0][0].length : 0) : 0) : 0) : 0) : 0,
  ]).filter(e => e != false).filter(e => e != undefined)
};