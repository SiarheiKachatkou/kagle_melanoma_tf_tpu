import tensorflow as tf

keys_tensor = tf.constant([0,1])
vals_tensor = tf.constant([1.0,2.0])
table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

def get_num_of_repetition_for_example(training_example):
    _, label = training_example
    label_int=tf.cast(label,tf.int32)
    num_to_repeat = table.lookup(label_int)
    #num_to_repeat_integral = tf.cast(int(num_to_repeat), tf.float32)
    #residue = num_to_repeat - num_to_repeat_integral
    #TODO residue?
    #num_to_repeat = num_to_repeat_integral + tf.cast(tf.random.uniform(shape=()) <= residue, tf.float32)

    return tf.cast(num_to_repeat, tf.int64)

def oversample(dataset,config):
    if config.oversample_mult!=1:
        dataset = dataset.flat_map(
            lambda input_data, label: tf.data.Dataset.from_tensors((input_data, label)).repeat(
                get_num_of_repetition_for_example((input_data, label))))

    return dataset