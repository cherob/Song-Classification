module.exports = {
    map: (x, i1, i2, o1, o2) => ((x - i1) * (o2 - o1)) / (i2 - i1) + o1,
    T: (A) => A[0].map((col, iii) => A.map(row => row[iii])),
    S: (A) => ([A.length || 0,
        A ? (A[0] ? A[0].length : 0) : 0,
        A ? (A[0] ? (A[0][0] ? A[0][0].length : 0) : 0) : 0,
        A ? (A[0] ? (A[0][0] ? (A[0][0][0] ? A[0][0][0].length : 0) : 0) : 0) : 0,
        A ? (A[0] ? (A[0][0] ? (A[0][0][0] ? (A[0][0][0][0] ? A[0][0][0][0].length : 0) : 0) : 0) : 0) : 0,
    ]).filter(e => e != false).filter(e => e != undefined),
    hamming: function (A, frame_length) {
        A = A.map(B => B.map(C => {
            C => C = C.map((value, index) => {
                let x = (2 * Math.PI * index) / (frame_length - 1);
                value *= 0.54 - 0.46 * Math.cos(x);
                return value
            });
            return C
        }));
        return A;
    },

    linspace: function (start, stop, num) {
        let c = [start];
        for (let i = 0; i < num - 1; i++) {
            c.push(c[i] + (stop - start) / (num + 1))
        }
        return c;
    },

    zeros: function (n) {
        let zerosArr = new Array(n).fill(0);
        return zerosArr
    }
}