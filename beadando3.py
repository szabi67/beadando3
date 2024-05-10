import tensorflow as tf

# Két tenzor létrehozása véletlen számokkal
tensor1 = tf.random.uniform((32, 10, 100, 3), minval = -10, maxval = 10)
tensor2 = tf.random.uniform((32, 10, 100, 3), minval = -10, maxval = 10)

# A távolságok kiszámítása
def euklides_tavolsag(tensor1, tensor2):
    # A távolságok kiszámítása az utolsó dimenzió mentén
    kulonbseg_negyzet = tf.square(tensor1 - tensor2)
    osszegzes = tf.reduce_sum(kulonbseg_negyzet, axis=-1)
    tavolsag = tf.sqrt(osszegzes)
    return tavolsag

# Távolságok kiszámítása
tavolsagok = euklides_tavolsag(tf.expand_dims(tensor1, axis=-2), tf.expand_dims(tensor2, axis=-3))

# Az eredmény kiírása
print("Az eredmény alakja:", tavolsagok.shape)

#Megjegyzés: A megoldás során chatgpt-t használtam a megfelelő függvények kikeresésére
