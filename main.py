import datetime

def responder_pregunta(pregunta):
    pregunta = pregunta.lower()

    if 'hora' in pregunta:
        hora_actual = obtener_hora_actual()
        return "La hora actual es " + hora_actual
    elif 'día' in pregunta or 'fecha' in pregunta:
        fecha_actual = obtener_fecha_actual()
        return "La fecha actual es " + fecha_actual
    else:
        return "Lo siento, no puedo responder esa pregunta."

def obtener_hora_actual():
    hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
    return hora_actual

def obtener_fecha_actual():
    fecha_actual = datetime.date.today().strftime("%d/%m/%Y")
    return fecha_actual

# Ejemplo de uso
pregunta_usuario = input("Hazme una pregunta sobre la hora o la fecha: ")
respuesta = responder_pregunta(pregunta_usuario)
print(respuesta)