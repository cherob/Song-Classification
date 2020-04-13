const numpy = require('numjs');

numpy.tile = (A, reps) => {
    let Av_ = reps[1] || false;
    let A_ = A
    let A__ = []

    
    for (let i = 0; i < reps; i++) {
        A.push(A_)
    }

    return A_;
};

module.exports = {
    preemphasis: function (signal, coeff = 0.95) {
        numpy.append(signal[0], signal.slice(1) - coeff * signal(0, -1));
    },
    framesig: function (sig, frame_len, frame_step, winfunc) {
        slen = sig.length;
        frame_len = int(Math.round(frame_len))
        frame_step = int(Math.round(frame_step))
        if (slen <= frame_len)
            numframes = 1
        else
            numframes = 1 + Math.ceil((1.0 * slen - frame_len) / frame_step)

        padlen = (numframes - 1) * frame_step + frame_len

        zeros = numpy.zeros(padlen - slen)
        padsignal = numpy.concatenate(sig, zeros)

        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1))
        indices = indices[0].map((col, iii) => indices.map(row => row[iii]));

        indices = num
        py.array(indices, dtype = numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))
        return frames * win
    }

}