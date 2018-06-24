from math import floor, ceil
import numpy as np

import tensorflow as tf

import mir_eval
mir_eval.multipitch.MIN_FREQ = 1

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, create_model, create_summaries = None):
        with self.session.graph.as_default():
            self.logdir = args["logdir"]
            self.note_range = args["note_range"]
            self.annotations_per_window = args["annotations_per_window"]
            self.window_size = args["window_size"]

            # Inputs
            self.window = tf.placeholder(tf.float32, [None, self.window_size], name="window")
            self.annotations = tf.placeholder(tf.int32, [None, self.annotations_per_window, None], name="annotations")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            self.ref_notes = tf.reduce_sum(tf.one_hot(self.annotations, self.note_range), axis=2)
            # wrong_labels = 1-tf.reduce_sum(tf.one_hot(tf.zeros(tf.shape(self.annotations), dtype=tf.int32), self.note_range), axis=2)
            dims = tf.shape(self.ref_notes)
            self.ref_notes = tf.concat([tf.zeros([dims[0], dims[1], 1]), self.ref_notes[:,:,1:]], axis=2) # annotations are padded with zeros, so we manually remove this additional note
            
            # create_model needs to provide these values:
            self.est_notes = None
            self.loss = None
            self.training = None

            create_model(self, args)

            assert self.est_notes != None
            assert self.loss != None
            assert self.training != None

            # self.est_notes
            ref_notes_b = tf.cast(self.ref_notes, tf.bool)
            est_notes_b = tf.cast(self.est_notes, tf.bool)
            true_positive_sum = tf.count_nonzero(tf.logical_and(ref_notes_b, est_notes_b))
            n_ref_sum = tf.count_nonzero(ref_notes_b)
            n_est_sum = tf.count_nonzero(est_notes_b)

            def safe_div(numerator, denominator):
                return tf.cond(denominator > 0, lambda: numerator/denominator, lambda: tf.constant(0, dtype=tf.float64))

            self.precision = safe_div(true_positive_sum, n_est_sum)
            self.recall = safe_div(true_positive_sum, n_ref_sum)
            acc_denom = n_est_sum + n_ref_sum - true_positive_sum
            self.accuracy = safe_div(true_positive_sum, acc_denom)
            
            # self.precision = tf.Print(self.precision, [self.annotations, self.ref_notes, true_positive_sum, n_ref_sum, n_est_sum, ref_notes_b, est_notes_b], summarize=10000)

            summary_writer = tf.contrib.summary.create_file_writer(self.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(200):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/precision", self.precision),
                                           tf.contrib.summary.scalar("train/recall", self.recall),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           ]


            # middle = int(self.annotations_per_window/2)
            # self.predictions_whole = tf.argmax(note_probs, axis=2)
            # self.predictions = tf.argmax(note_probs[:,middle], axis=1)
             
            # Summaries
            """
            with tf.name_scope("metrics"):
                ref = self.annotations[:,middle]
                est = self.predictions

                self.accuracy_whole = tf.reduce_mean(tf.cast(tf.equal(self.annotations, self.predictions_whole), tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(ref, est), tf.float32))

                unvoiced_ref = tf.equal(ref, 0)
                voiced_ref = tf.logical_not(unvoiced_ref)
                unvoiced_est = tf.equal(est, 0)
                voiced_est = tf.logical_not(unvoiced_est)

                unvoiced_sum = tf.count_nonzero(unvoiced_ref)
                voiced_sum = tf.count_nonzero(voiced_ref)

                unvoiced_TP = tf.count_nonzero(tf.logical_and(voiced_ref, voiced_est))
                unvoiced_FP = tf.count_nonzero(tf.logical_and(unvoiced_ref, voiced_est))
                # proportion of frames labeled as melody frames in the reference that are estimated as melody frames
                with tf.name_scope("recall"):
                    recall = unvoiced_TP/tf.maximum(voiced_sum, 1)  # min. 1 to avoid division by zero
                # proportion of frames labeled as non-melody in the reference that are mistakenly estimated as melody frames
                with tf.name_scope("false_alarm"):
                    false_alarm = unvoiced_FP/tf.maximum(unvoiced_sum, 1)  # min. 1 to avoid division by zero

                # print(voiced_ref.dtype, voiced_ref.shape)
                # precision = tf.Print(false_alarm, [(ref), (voiced_ref)])
                voiced_ref_notes_ref = tf.boolean_mask(ref, voiced_ref)
                voiced_est_notes_est = tf.boolean_mask(est, voiced_ref)
                raw_pitch = tf.count_nonzero(tf.equal(voiced_ref_notes_ref, voiced_est_notes_est))/tf.maximum(voiced_sum, 1)
                raw_chroma = tf.count_nonzero(tf.equal(tf.mod(voiced_ref_notes_ref, 12), tf.mod(voiced_est_notes_est, 12)))/tf.maximum(voiced_sum, 1)

            summary_writer = tf.contrib.summary.create_file_writer(self.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(200):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy_whole", self.accuracy_whole),
                                           tf.contrib.summary.scalar("train/recall", recall),
                                           tf.contrib.summary.scalar("train/false_alarm", false_alarm),
                                           tf.contrib.summary.scalar("train/raw_pitch", raw_pitch),
                                           tf.contrib.summary.scalar("train/raw_chroma", raw_chroma),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.summaries["train_detailed"] = []
                batch_estimation = draw_notes(ref, est, 4, 2.36, ".")
                self.summaries["train_detailed"].append(tf.contrib.summary.image('train/batch_estimation', batch_estimation))
                
                example_estimation = draw_notes(self.annotations[0], self.predictions_whole[0], 4, 2.36, ".")
                self.summaries["train_detailed"].append(tf.contrib.summary.image('train/example_estimation', example_estimation))
                
                image_spectrogram = draw_spectrum(self.window[0], args["samplerate"])
                self.summaries["train_detailed"].append(tf.contrib.summary.image('train/example_spectrogram', image_spectrogram))
                
                self.summaries["train_detailed"].append(tf.contrib.summary.audio('train/example_batch_audio', self.window, args["samplerate"], 1))
                
                # out1 = [tf.nn.sigmoid(tf.expand_dims(tf.transpose(audio_net[0])[::-1], 2))]
                out1 = [tf.expand_dims(tf.nn.sigmoid(tf.transpose(note_probs[0])[::-1]), 2)]
                self.summaries["train_detailed"].append(tf.contrib.summary.image('train/example_audio_net_output', out1))
                # self.summaries["train_detailed"].append(tf.contrib.summary.image('train/example_audio_net_ears_output', out2))
                # out2 = [tf.nn.sigmoid(tf.expand_dims(tf.transpose(audio_net_ears[0])[::-1], 2))]
                
                # for i,l in enumerate(self.anet_layers):
                #    self.summaries["train_detailed"].append(tf.contrib.summary.histogram("train/layer_"+str(i+1), l))
                
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                self.given_recall = tf.placeholder(tf.float32, [], name="given_recall")
                self.given_false_alarm = tf.placeholder(tf.float32, [], name="given_false_alarm")
                self.given_raw_pitch = tf.placeholder(tf.float32, [], name="given_raw_pitch")
                self.given_raw_chroma = tf.placeholder(tf.float32, [], name="given_raw_chroma")

                self.given_notes_ref = tf.placeholder(tf.int16, [None], name="given_notes_ref")
                self.given_notes_est = tf.placeholder(tf.int16, [None], name="given_notes_est")

                image_tensor_overview = draw_notes(self.given_notes_ref, self.given_notes_est, 18, 2.36, ",")
                image_tensor_detail = draw_notes(self.given_notes_ref, self.given_notes_est, 200, 2.355*2, ",")
                
                self.summaries["test"] = [tf.contrib.summary.scalar("test/loss", self.given_loss),
                                          tf.contrib.summary.scalar("test/accuracy", self.given_accuracy),
                                          tf.contrib.summary.scalar("test/recall", self.given_recall),
                                          tf.contrib.summary.scalar("test/false_alarm", self.given_false_alarm),
                                          tf.contrib.summary.scalar("test/raw_pitch", self.given_raw_pitch),
                                          tf.contrib.summary.scalar("test/raw_chroma", self.given_raw_chroma),
                                          tf.contrib.summary.image('test/estimation_overview', image_tensor_overview),
                                          tf.contrib.summary.image('test/estimation_detail', image_tensor_detail),
                                         ]
            
            if create_summaries:
                create_summaries(self, args, summary_writer)
            """
            
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def _unpack_one(self, batch, key):
        return [b[key] for b in batch]
    def _unpack_batch(self, batch):
        return self._unpack_one(batch, "audio"), self._unpack_one(batch, "annotation")

    def train(self, batch):
        windows, annotations = self._unpack_batch(batch)

        out = self.session.run([self.accuracy, self.loss, self.training, self.summaries["train"]],
                         {self.window: windows, self.annotations: annotations, self.is_training: True})
        return out[0], out[1]
    def train_detailed(self, batch):
        windows, annotations = self._unpack_batch(batch)
        batch_annotations_ragged = self._unpack_one(batch, "annotation_ragged")

        out = self.session.run([self.accuracy, self.loss, self.est_notes,
            self.precision, self.recall, self.training, self.summaries["train"]],
                         {self.window: windows, self.annotations: annotations, self.is_training: True})

        # print(out[3], out[4], out[0])
        # est_annotation = [[i for i, v in enumerate(est_notes_frame) if v == 1] for est_notes in out[2] for est_notes_frame in est_notes]
        # ref_annotation = [est_notes_frame for est_notes in batch_annotations_ragged for est_notes_frame in est_notes]

        # ref = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in ref_annotation])
        # est = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in est_annotation])
        # t = np.arange(0, len(ref_annotation))*0.01

        # metrics = mir_eval.multipitch.metrics(t, ref, t, est)
        # print(metrics)

        return out[0], out[1]
    
    def evaluate(self, dataset, batch_size):
        dataset.reset()
        estimation = []
        reference = []
        loss = 0
        c = 0
        while not dataset.epoch_finished():
            batch = dataset.next_batch(batch_size)
            batch_windows, batch_annotations = self._unpack_batch(batch)
            batch_annotations_ragged = self._unpack_one(batch, "annotation_ragged")

            batch_est_notes, batch_loss = self.session.run([self.est_notes, self.loss],
                {self.window: batch_windows, self.annotations: batch_annotations, self.is_training: False})

            est_annotation = [[i for i, v in enumerate(est_notes_frame) if v == 1] for est_notes in batch_est_notes for est_notes_frame in est_notes]
            ref_annotation = [est_notes_frame for est_notes in batch_annotations_ragged for est_notes_frame in est_notes]

            assert len(est_annotation) == len(ref_annotation)
            estimation.append(est_annotation)
            reference.append(ref_annotation)

            loss += batch_loss * len(batch_windows)
            c += len(batch_windows)
        loss /= c

        ref = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for batch in reference for notes in batch])
        est = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for batch in estimation for notes in batch])
        t = np.arange(0, len(ref))*0.01

        assert len(ref) == len(est)

        metrics = mir_eval.multipitch.metrics(t, ref, t, est)
        print("eval",metrics)

        # print("generating summaries")
        # self.session.run(self.summaries["test"], {
        #     self.given_loss: loss,
        #     self.given_accuracy: overall_accuracy,
        #     self.given_recall: recall,
        #     self.given_false_alarm: false_alarm,
        #     self.given_raw_pitch: raw_pitch,
        #     self.given_raw_chroma: raw_chroma,
        #     self.given_notes_est: est,
        #     self.given_notes_ref: ref
        # })
        # print("done")

        return metrics[2]

    def predict(self, dataset, batch_size, skip=0, batches=None):
        labels = []
        while not dataset.epoch_finished():
            batch_windows, _ = dataset.next_batch(batch_size)
            if skip > 0:
                skip -= 1
                continue

            pred = self.session.run(self.predictions,
                {self.window: batch_windows, self.is_training: False})
            labels.append(pred)

            if batches != None:
                batches -= 1
                if batches == 0:
                    break
                
        return np.concatenate(labels)
    
    def save(self):
        save_path = self.saver.save(self.session, self.logdir+"/model.ckpt")
        print("Model saved in path:", save_path)
    def restore(self):
        self.saver.restore(self.session, self.logdir+"/model.ckpt")
        print("Model restored from path:", self.logdir)