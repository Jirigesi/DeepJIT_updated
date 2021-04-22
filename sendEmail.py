import smtplib, ssl

def sendEmail(dirpath:str):
    port = 465  # For SSL

    smtp_server = "smtp.gmail.com"
    sender_email = "runmodel.jiri@gmail.com"  # Enter your address
    password = "JIrigesi3355"

    receiver_email = "jirigesi@gmail.com"  # Enter receiver address
    subject = "Model Training Complete"
    body = "Hey, your model training is finished.\n\n It is saved in path:\n " + dirpath

    message = """\
        Subject: %s

        %s
        """ % (subject, body)


    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login("runmodel.jiri@gmail.com", password)
        print("Login successfully...")
        server.sendmail(sender_email, receiver_email, message)
        print("Email sent successfully...")

if __name__ == '__main__':
    dirpath = "a/b/c"

    sendEmail(dirpath)


