from tensorflow.keras.models import load_model

from main import X_test

model = load_model("mlp_flight_delay_model.h5")

model.summary()

y_pred = model.predict(X_test)

