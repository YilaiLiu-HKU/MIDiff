image=0.8
gamma = 0.25
image = np.power(np.abs(image), gamma) * np.sign(image)
image=(image+1)/2