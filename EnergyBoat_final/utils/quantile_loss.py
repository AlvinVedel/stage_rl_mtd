import tensorflow as tf

class QuantileHuberLoss(tf.keras.losses.Loss):
    def __init__(self, taus, kappa=1.0, name="quantile_huber_loss"):
        super(QuantileHuberLoss, self).__init__(name=name)
        self.taus = tf.constant(taus, dtype=tf.float32)  # Assurez-vous que taus est bien un tensor
        self.kappa = kappa

    def call(self, y_true, y_pred):
        # Calcul de l'erreur
        error = y_true - y_pred
        # Calcul de la perte de Huber
        huber_loss = tf.where(
            tf.abs(error) <= self.kappa,
            0.5 * tf.square(error),
            self.kappa * (tf.abs(error) - 0.5 * self.kappa)
        )
        # Redimensionner tau pour qu'il soit broadcastable avec l'erreur
        taus = tf.reshape(self.taus, [1, -1])
        # Calcul de la Quantile Huber Loss
        quantile_loss = tf.abs(taus - tf.cast(error < 0.0, tf.float32)) * huber_loss        
        # Moyenne de la perte sur les quantiles et les batchs
        return tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=-1))