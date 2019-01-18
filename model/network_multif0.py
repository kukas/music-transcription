class NetworkMultif0(Network):
    def _summaries(self, args, summary_writer):
        # batch metrics
        with tf.name_scope("metrics"):
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

        self.summaries = {}
        with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(200):
            self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                        tf.contrib.summary.scalar("train/precision", self.precision),
                                        tf.contrib.summary.scalar("train/recall", self.recall),
                                        tf.contrib.summary.scalar("train/accuracy", self.accuracy), ]

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
            self.given_precision = tf.placeholder(tf.float32, [], name="given_precision")
            self.given_recall = tf.placeholder(tf.float32, [], name="given_recall")
            self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
            self.given_e_sub = tf.placeholder(tf.float32, [], name="given_e_sub")
            self.given_e_miss = tf.placeholder(tf.float32, [], name="given_e_miss")
            self.given_e_fa = tf.placeholder(tf.float32, [], name="given_e_fa")
            self.given_e_tot = tf.placeholder(tf.float32, [], name="given_e_tot")
            self.given_precision_chroma = tf.placeholder(tf.float32, [], name="given_precision_chroma")
            self.given_recall_chroma = tf.placeholder(tf.float32, [], name="given_recall_chroma")
            self.given_accuracy_chroma = tf.placeholder(tf.float32, [], name="given_accuracy_chroma")
            self.given_e_sub_chroma = tf.placeholder(tf.float32, [], name="given_e_sub_chroma")
            self.given_e_miss_chroma = tf.placeholder(tf.float32, [], name="given_e_miss_chroma")
            self.given_e_fa_chroma = tf.placeholder(tf.float32, [], name="given_e_fa_chroma")
            self.given_e_tot_chroma = tf.placeholder(tf.float32, [], name="given_e_tot_chroma")

            self.image1 = tf.placeholder(tf.uint8, [None, None, 4], name="image1")
            image1 = tf.expand_dims(self.image1, 0)

            self.summaries["test_small"] = [tf.contrib.summary.image("test_small/image1", image1),
                                            tf.contrib.summary.scalar("test_small/loss", self.given_loss),
                                            tf.contrib.summary.scalar("test_small/precision", self.given_precision),
                                            tf.contrib.summary.scalar("test_small/recall", self.given_recall),
                                            tf.contrib.summary.scalar("test_small/accuracy", self.given_accuracy), ]

            self.summaries["test"] = [tf.contrib.summary.scalar("test/loss", self.given_loss),
                                        tf.contrib.summary.scalar("test/precision", self.given_precision),
                                        tf.contrib.summary.scalar("test/recall", self.given_recall),
                                        tf.contrib.summary.scalar("test/accuracy", self.given_accuracy),
                                        tf.contrib.summary.scalar("test/e_sub", self.given_e_sub),
                                        tf.contrib.summary.scalar("test/e_miss", self.given_e_miss),
                                        tf.contrib.summary.scalar("test/e_fa", self.given_e_fa),
                                        tf.contrib.summary.scalar("test/e_tot", self.given_e_tot),
                                        tf.contrib.summary.scalar("test/precision_chroma", self.given_precision_chroma),
                                        tf.contrib.summary.scalar("test/recall_chroma", self.given_recall_chroma),
                                        tf.contrib.summary.scalar("test/accuracy_chroma", self.given_accuracy_chroma),
                                        tf.contrib.summary.scalar("test/e_sub_chroma", self.given_e_sub_chroma),
                                        tf.contrib.summary.scalar("test/e_miss_chroma", self.given_e_miss_chroma),
                                        tf.contrib.summary.scalar("test/e_fa_chroma", self.given_e_fa_chroma),
                                        tf.contrib.summary.scalar("test/e_tot_chroma", self.given_e_tot_chroma),
                                        # tf.contrib.summary.image('test/image1', self.image1),
                                        # tf.contrib.summary.image('test/estimation_detail', image_tensor_detail),
                                        ]

    def evaluate(self, dataset, batch_size, visual_output=False, print_detailed=False):
        reference, estimation, loss = self.predict(dataset, batch_size)

        ref = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in reference])
        est = np.array([mir_eval.util.midi_to_hz(np.array(notes)) for notes in estimation])
        t = np.arange(0, len(ref))*0.01

        metrics = mir_eval.multipitch.metrics(t, ref, t, est)
        # unpack metrics
        (given_precision, given_recall, given_accuracy,
         given_e_sub, given_e_miss, given_e_fa, given_e_tot,
         given_precision_c, given_recall_c, given_accuracy_c,
         given_e_sub_c, given_e_miss_c, given_e_fa_c, given_e_tot_c) = metrics

        if print_detailed:
            print("Precision", given_precision)
            print("Recall", given_recall)
            print("Accuracy", given_accuracy)
            print("Substitution Error", given_e_sub)
            print("Miss Error", given_e_miss)
            print("False Alarm Error", given_e_fa)
            print("Total Error", given_e_tot)
            print("Chroma Precision", given_precision_c)
            print("Chroma Recall", given_recall_c)
            print("Chroma Accuracy", given_accuracy_c)
            print("Chroma Substitution Error", given_e_sub_c)
            print("Chroma Miss Error", given_e_miss_c)
            print("Chroma False Alarm Error", given_e_fa_c)
            print("Chroma Total Error", given_e_tot_c)

        # write evaluation metrics to tf summary
        if visual_output:
            fig = vis.draw_notes(reference, estimation)
            image1 = vis.fig2data(fig)

            # suppress inline mode
            if not print_detailed:
                plt.close()

            self.session.run(self.summaries["test_small"], {
                self.image1: image1,
                self.given_loss: loss,
                self.given_precision: given_precision,
                self.given_recall: given_recall,
                self.given_accuracy: given_accuracy,
            })
        else:
            self.session.run(self.summaries["test"], {
                self.given_loss: loss,
                # mir_eval summary
                self.given_precision: given_precision,
                self.given_recall: given_recall,
                self.given_accuracy: given_accuracy,
                self.given_e_sub: given_e_sub,
                self.given_e_miss: given_e_miss,
                self.given_e_fa: given_e_fa,
                self.given_e_tot: given_e_tot,
                self.given_precision_chroma: given_precision_c,
                self.given_recall_chroma: given_recall_c,
                self.given_accuracy_chroma: given_accuracy_c,
                self.given_e_sub_chroma: given_e_sub_c,
                self.given_e_miss_chroma: given_e_miss_c,
                self.given_e_fa_chroma: given_e_fa_c,
                self.given_e_tot_chroma: given_e_tot_c,
            })

        return given_accuracy