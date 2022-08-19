import numpy as np

import keras
from keras.models import Sequential

def generate_network():
    #[a,b] a = 0 -> husband(you) wrong, a = 1 -> husband(you) wrong ; b = 0 -> wife wrong, b = 1 -> wife right
    #all possible entry
    table=np.array([[1,1],[1,0],[0,1],[0,0]])

    #[a,b] a = 0 -> husband(you), a = 1 -> wife ; b = 0 -> wrong b = 1 -> right
    #all possible output in good order compared to table(entry)
    logic=np.array([[1,1],[1,1],[1,1],[0,0]])

    from keras.layers import Dense

    network = Sequential()

    network.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    network.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'sigmoid'))
    network.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'sigmoid'))
    network.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))

    network.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    network.fit(table, logic, batch_size = 4, epochs = 3000)

    network.save("mariage_logic_gate.h5")

def who_is_right_or_wrong(husband, wife):
    if (husband == 0 or husband == 1) and (wife == 0 or wife == 1):
        network = keras.models.load_model("mariage_logic_gate.h5")
        result = network.predict(np.array([[husband,wife]]))
        answer = ""
        if result[0][0]<0.5:
            answer += "husband is "
        else:
            answer += "wife is "
        if result[0][1]<0.5:
            answer += "wrong"
        else:
            answer += "right"
        print(answer)
    else:
        print("husband and wife value must be equals to 0 or 1")
        return
