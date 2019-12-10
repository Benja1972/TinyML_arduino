echo "const unsigned char model[] = {" > model.h
cat gesture_model.tflite | xxd -i      >> model.h
echo "};"                              >> model.h
