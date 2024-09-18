import re
import matplotlib.pyplot as plt

# Input string
data = """
Training dataset: FMA-C-1-fixed-SCNN-Train.h5 + SCNN-Jamendo-train.h5
Training dataset length: 25611
Validation dataset length: 4085
Using device: cuda
Epoch 1:
Train loss: 119.72003811597824, Train accuracy: 52.99%
Validation loss: 0.71729530, Validation accuracy: 40.86%

Epoch 2:
Train loss: 118.36904788017273, Train accuracy: 52.98%
Validation loss: 0.71931467, Validation accuracy: 40.86%

Epoch 3:
Train loss: 119.5847098827362, Train accuracy: 55.38%
Validation loss: 0.74837617, Validation accuracy: 40.86%

Epoch 4:
Train loss: 123.02831661701202, Train accuracy: 55.68%
Validation loss: 0.71109280, Validation accuracy: 40.86%

Epoch 5:
Train loss: 120.21534380316734, Train accuracy: 55.20%
Validation loss: 0.89143904, Validation accuracy: 40.88%

Epoch 6:
Train loss: 116.4517533481121, Train accuracy: 58.23%
Validation loss: 0.91082734, Validation accuracy: 40.88%

Epoch 7:
Train loss: 117.01944816112518, Train accuracy: 58.99%
Validation loss: 0.71564711, Validation accuracy: 40.95%

Epoch 8:
Train loss: 114.28647819161415, Train accuracy: 60.20%
Validation loss: 0.76831180, Validation accuracy: 40.86%

Epoch 9:
Train loss: 116.27589051425457, Train accuracy: 61.65%
Validation loss: 0.72462808, Validation accuracy: 40.95%

Epoch 10:
Train loss: 115.26931002736092, Train accuracy: 61.77%
Validation loss: 1.00664394, Validation accuracy: 41.69%

Epoch 11:
Train loss: 110.1529074087739, Train accuracy: 65.36%
Validation loss: 0.79477909, Validation accuracy: 41.69%

Epoch 12:
Train loss: 107.43564809858799, Train accuracy: 66.76%
Validation loss: 0.71104341, Validation accuracy: 42.01%

Epoch 13:
Train loss: 103.19810847938061, Train accuracy: 69.56%
Validation loss: 0.70144677, Validation accuracy: 50.40%

Epoch 14:
Train loss: 99.60929058492184, Train accuracy: 71.16%
Validation loss: 0.76912438, Validation accuracy: 57.75%

Epoch 15:
Train loss: 95.03681647777557, Train accuracy: 72.99%
Validation loss: 0.80593225, Validation accuracy: 56.13%

Epoch 16:
Train loss: 94.10227340459824, Train accuracy: 73.36%
Validation loss: 0.82076689, Validation accuracy: 55.28%

Epoch 17:
Train loss: 91.40138766169548, Train accuracy: 74.42%
Validation loss: 0.81916010, Validation accuracy: 56.21%

Epoch 18:
Train loss: 90.26407589018345, Train accuracy: 75.43%
Validation loss: 0.75325367, Validation accuracy: 55.10%

Epoch 19:
Train loss: 91.14755390584469, Train accuracy: 74.49%
Validation loss: 0.88799756, Validation accuracy: 54.88%

Epoch 20:
Train loss: 87.08695790916681, Train accuracy: 76.11%
Validation loss: 0.85133374, Validation accuracy: 55.01%

Epoch 21:
Train loss: 85.95909655094147, Train accuracy: 76.52%
Validation loss: 0.79351459, Validation accuracy: 57.82%

Epoch 22:
Train loss: 85.5301416516304, Train accuracy: 76.89%
Validation loss: 0.81619297, Validation accuracy: 60.17%

Epoch 23:
Train loss: 84.78537954390049, Train accuracy: 76.83%
Validation loss: 0.80688310, Validation accuracy: 60.29%

Epoch 24:
Train loss: 85.02497211098671, Train accuracy: 76.80%
Validation loss: 0.80145352, Validation accuracy: 59.19%

Epoch 25:
Train loss: 83.76488952338696, Train accuracy: 77.24%
Validation loss: 0.79040339, Validation accuracy: 59.12%

Epoch 26:
Train loss: 84.01812583208084, Train accuracy: 77.08%
Validation loss: 0.76527710, Validation accuracy: 60.83%

Epoch 27:
Train loss: 84.74702878296375, Train accuracy: 76.74%
Validation loss: 0.87007594, Validation accuracy: 58.48%

Epoch 28:
Train loss: 86.90082956850529, Train accuracy: 75.96%
Validation loss: 0.84092407, Validation accuracy: 55.59%

Epoch 29:
Train loss: 90.10583382844925, Train accuracy: 74.26%
Validation loss: 0.84848947, Validation accuracy: 58.36%

Epoch 30:
Train loss: 90.6935382783413, Train accuracy: 73.86%
Validation loss: 0.91082687, Validation accuracy: 59.29%

Epoch 31:
Train loss: 88.8367508649826, Train accuracy: 75.10%
Validation loss: 0.88148603, Validation accuracy: 60.81%

Epoch 32:
Train loss: 90.15317545831203, Train accuracy: 74.50%
Validation loss: 0.90520473, Validation accuracy: 60.27%

Epoch 33:
Train loss: 91.59909997880459, Train accuracy: 73.75%
Validation loss: 0.84634822, Validation accuracy: 59.66%

Epoch 34:
Train loss: 92.01243840157986, Train accuracy: 73.64%
Validation loss: 0.79732994, Validation accuracy: 61.59%

Epoch 35:
Train loss: 93.57427275180817, Train accuracy: 73.60%
Validation loss: 0.92522562, Validation accuracy: 59.98%

Epoch 36:
Train loss: 89.72577413916588, Train accuracy: 74.77%
Validation loss: 0.89131693, Validation accuracy: 61.57%

Epoch 37:
Train loss: 87.55549770593643, Train accuracy: 75.58%
Validation loss: 0.91403501, Validation accuracy: 60.15%

Epoch 38:
Train loss: 88.83500169962645, Train accuracy: 75.17%
Validation loss: 0.86488567, Validation accuracy: 62.64%

Epoch 39:
Train loss: 87.46049053221941, Train accuracy: 75.46%
Validation loss: 0.93620806, Validation accuracy: 59.17%

Epoch 40:
Train loss: 86.60406413674355, Train accuracy: 76.28%
Validation loss: 0.77039536, Validation accuracy: 62.55%

Epoch 41:
Train loss: 85.98654481768608, Train accuracy: 75.98%
Validation loss: 0.82612060, Validation accuracy: 62.72%

Epoch 42:
Train loss: 85.1938194707036, Train accuracy: 76.42%
Validation loss: 0.76606599, Validation accuracy: 60.76%

Epoch 43:
Train loss: 86.26980410516262, Train accuracy: 76.10%
Validation loss: 0.76098417, Validation accuracy: 63.06%

Epoch 44:
Train loss: 85.4320569485426, Train accuracy: 76.38%
Validation loss: 0.77695526, Validation accuracy: 63.45%

Epoch 45:
Train loss: 85.90369316935539, Train accuracy: 76.23%
Validation loss: 0.80358531, Validation accuracy: 62.57%

Epoch 46:
Train loss: 85.04972243309021, Train accuracy: 76.63%
Validation loss: 0.79510396, Validation accuracy: 61.15%

Epoch 47:
Train loss: 86.21332271397114, Train accuracy: 76.09%
Validation loss: 0.76542434, Validation accuracy: 61.47%

Epoch 48:
Train loss: 86.30825337022543, Train accuracy: 75.79%
Validation loss: 0.79968351, Validation accuracy: 62.40%

Epoch 49:
Train loss: 85.82722739875317, Train accuracy: 76.12%
Validation loss: 0.82142170, Validation accuracy: 60.47%

Epoch 50:
Train loss: 84.11239697039127, Train accuracy: 76.67%
Validation loss: 0.75550961, Validation accuracy: 63.23%

Epoch 51:
Train loss: 85.60248044878244, Train accuracy: 76.22%
Validation loss: 0.81102466, Validation accuracy: 59.56%

Epoch 52:
Train loss: 86.67589135468006, Train accuracy: 75.73%
Validation loss: 0.72237324, Validation accuracy: 65.24%

Epoch 53:
Train loss: 86.08983813226223, Train accuracy: 75.99%
Validation loss: 0.73581746, Validation accuracy: 64.21%

Epoch 54:
Train loss: 84.08447223901749, Train accuracy: 76.64%
Validation loss: 0.73623808, Validation accuracy: 63.94%

Epoch 55:
Train loss: 87.16493057459593, Train accuracy: 75.58%
Validation loss: 0.74821485, Validation accuracy: 62.52%

Epoch 56:
Train loss: 88.58593101799488, Train accuracy: 75.23%
Validation loss: 0.79521464, Validation accuracy: 61.59%

Epoch 57:
Train loss: 86.52742307633162, Train accuracy: 75.37%
Validation loss: 0.89995087, Validation accuracy: 58.07%

Epoch 58:
Train loss: 87.62562599778175, Train accuracy: 75.08%
Validation loss: 0.84152242, Validation accuracy: 58.63%

Epoch 59:
Train loss: 87.41187863051891, Train accuracy: 75.58%
Validation loss: 0.87188404, Validation accuracy: 58.26%

Epoch 60:
Train loss: 87.39037631452084, Train accuracy: 75.48%
Validation loss: 0.82566198, Validation accuracy: 59.73%

Epoch 61:
Train loss: 85.72929358482361, Train accuracy: 75.87%
Validation loss: 0.81786515, Validation accuracy: 61.79%

Epoch 62:
Train loss: 84.75666670501232, Train accuracy: 76.08%
Validation loss: 0.71506646, Validation accuracy: 65.83%

Epoch 63:
Train loss: 85.69213904440403, Train accuracy: 76.40%
Validation loss: 0.77701108, Validation accuracy: 64.41%

Epoch 64:
Train loss: 87.56332887709141, Train accuracy: 75.49%
Validation loss: 0.79827960, Validation accuracy: 63.77%

Epoch 65:
Train loss: 87.13505965471268, Train accuracy: 75.57%
Validation loss: 0.75046928, Validation accuracy: 64.43%

Epoch 66:
Train loss: 85.95008346438408, Train accuracy: 75.92%
Validation loss: 0.73954378, Validation accuracy: 65.31%

Epoch 67:
Train loss: 85.63760125637054, Train accuracy: 76.11%
Validation loss: 0.76491267, Validation accuracy: 63.79%

Epoch 68:
Train loss: 86.60420123487711, Train accuracy: 75.80%
Validation loss: 0.71676128, Validation accuracy: 63.70%

Epoch 69:
Train loss: 88.74730734527111, Train accuracy: 74.72%
Validation loss: 0.78726412, Validation accuracy: 61.57%

Epoch 70:
Train loss: 87.81955511868, Train accuracy: 74.98%
Validation loss: 0.79446070, Validation accuracy: 65.41%

Epoch 71:
Train loss: 88.7276518791914, Train accuracy: 75.26%
Validation loss: 0.74643955, Validation accuracy: 61.30%

Epoch 72:
Train loss: 90.3738794028759, Train accuracy: 74.01%
Validation loss: 0.80133842, Validation accuracy: 58.95%

Epoch 73:
Train loss: 91.87628492712975, Train accuracy: 73.60%
Validation loss: 0.81271023, Validation accuracy: 61.47%

Epoch 74:
Train loss: 95.10818433761597, Train accuracy: 72.74%
Validation loss: 0.82791652, Validation accuracy: 59.34%

Epoch 75:
Train loss: 91.56311854720116, Train accuracy: 73.68%
Validation loss: 0.83730466, Validation accuracy: 61.79%

Epoch 76:
Train loss: 92.03162385523319, Train accuracy: 73.51%
Validation loss: 0.73949728, Validation accuracy: 62.72%

Epoch 77:
Train loss: 90.36526983976364, Train accuracy: 74.02%
Validation loss: 0.84076269, Validation accuracy: 61.22%

Epoch 78:
Train loss: 89.52110607922077, Train accuracy: 74.37%
Validation loss: 0.78035674, Validation accuracy: 61.22%

Epoch 79:
Train loss: 89.2955147176981, Train accuracy: 74.51%
Validation loss: 0.75727804, Validation accuracy: 62.35%

Epoch 80:
Train loss: 89.9782573133707, Train accuracy: 74.18%
Validation loss: 0.84167596, Validation accuracy: 60.05%

Epoch 81:
Train loss: 90.17591488361359, Train accuracy: 74.08%
Validation loss: 0.81007263, Validation accuracy: 61.64%

Epoch 82:
Train loss: 89.28319561481476, Train accuracy: 74.38%
Validation loss: 0.88251883, Validation accuracy: 61.66%

Epoch 83:
Train loss: 86.9623254686594, Train accuracy: 75.33%
Validation loss: 0.82917182, Validation accuracy: 63.13%

Epoch 84:
Train loss: 87.1396866440773, Train accuracy: 75.17%
Validation loss: 0.77537345, Validation accuracy: 64.06%

Epoch 85:
Train loss: 85.45346154272556, Train accuracy: 75.90%
Validation loss: 0.75396272, Validation accuracy: 63.45%

Epoch 86:
Train loss: 85.14964035153389, Train accuracy: 76.19%
Validation loss: 0.73783717, Validation accuracy: 63.99%

Epoch 87:
Train loss: 83.87593495845795, Train accuracy: 76.61%
Validation loss: 0.86330152, Validation accuracy: 62.06%

Epoch 88:
Train loss: 83.90651628375053, Train accuracy: 76.62%
Validation loss: 0.84494946, Validation accuracy: 62.23%

Epoch 89:
Train loss: 83.97027894854546, Train accuracy: 76.64%
Validation loss: 0.77497494, Validation accuracy: 63.08%

Epoch 90:
Train loss: 85.87893109023571, Train accuracy: 76.42%
Validation loss: 0.77128088, Validation accuracy: 63.06%

Epoch 91:
Train loss: 84.41981131583452, Train accuracy: 76.76%
Validation loss: 0.84482426, Validation accuracy: 62.08%

Epoch 92:
Train loss: 85.25159552693367, Train accuracy: 76.16%
Validation loss: 0.84352376, Validation accuracy: 61.52%

Epoch 93:
Train loss: 89.08537864685059, Train accuracy: 74.77%
Validation loss: 0.74087509, Validation accuracy: 62.35%

Epoch 94:
Train loss: 87.73144364356995, Train accuracy: 74.96%
Validation loss: 0.85075904, Validation accuracy: 61.49%

Epoch 95:
Train loss: 89.51681415736675, Train accuracy: 74.60%
Validation loss: 0.82570609, Validation accuracy: 62.59%

Epoch 96:
Train loss: 88.72902259230614, Train accuracy: 74.72%
Validation loss: 0.79396756, Validation accuracy: 63.16%

Epoch 97:
Train loss: 90.30536578595638, Train accuracy: 74.13%
Validation loss: 0.87372476, Validation accuracy: 61.57%

Epoch 98:
Train loss: 90.67776681482792, Train accuracy: 73.92%
Validation loss: 0.86825691, Validation accuracy: 62.01%

Epoch 99:
Train loss: 91.60221263766289, Train accuracy: 73.74%
Validation loss: 0.79779128, Validation accuracy: 61.18%

Epoch 100:
Train loss: 89.34728121757507, Train accuracy: 74.19%
Validation loss: 0.86231828, Validation accuracy: 61.18%

Epoch 101:
Train loss: 87.48351982235909, Train accuracy: 75.39%
Validation loss: 0.85125245, Validation accuracy: 62.91%

Epoch 102:
Train loss: 86.34395581483841, Train accuracy: 75.72%
Validation loss: 0.77167529, Validation accuracy: 64.26%

Epoch 103:
Train loss: 86.28876283764839, Train accuracy: 75.94%
Validation loss: 0.85045038, Validation accuracy: 61.49%

Epoch 104:
Train loss: 87.3628001511097, Train accuracy: 75.19%
Validation loss: 0.94657475, Validation accuracy: 59.98%

Epoch 105:
Train loss: 86.0082346946001, Train accuracy: 75.67%
Validation loss: 0.91935673, Validation accuracy: 62.18%

Epoch 106:
Train loss: 85.03297778964043, Train accuracy: 76.20%
Validation loss: 0.85639611, Validation accuracy: 62.79%

Epoch 107:
Train loss: 85.8598395138979, Train accuracy: 75.93%
Validation loss: 0.78299057, Validation accuracy: 63.50%

Epoch 108:
Train loss: 85.6474853605032, Train accuracy: 75.82%
Validation loss: 0.76708410, Validation accuracy: 64.41%

Epoch 109:
Train loss: 84.31113281846046, Train accuracy: 76.47%
Validation loss: 0.82662619, Validation accuracy: 63.35%

Epoch 110:
Train loss: 83.73782931268215, Train accuracy: 76.59%
Validation loss: 0.81585780, Validation accuracy: 63.53%

Epoch 111:
Train loss: 82.42581836879253, Train accuracy: 77.21%
Validation loss: 0.86911369, Validation accuracy: 62.28%

Epoch 112:
Train loss: 82.65479892492294, Train accuracy: 77.15%
Validation loss: 0.86292183, Validation accuracy: 62.25%

Epoch 113:
Train loss: 82.1971926689148, Train accuracy: 77.19%
Validation loss: 0.82916172, Validation accuracy: 63.77%

Epoch 114:
Train loss: 80.68912908434868, Train accuracy: 77.70%
Validation loss: 0.81608631, Validation accuracy: 64.80%

Epoch 115:
Train loss: 83.5033271163702, Train accuracy: 76.83%
Validation loss: 0.79566318, Validation accuracy: 64.87%

Epoch 116:
Train loss: 81.83753208816051, Train accuracy: 77.32%
Validation loss: 0.80082265, Validation accuracy: 64.36%

Epoch 117:
Train loss: 82.0101708471775, Train accuracy: 77.33%
Validation loss: 0.82991872, Validation accuracy: 63.82%

Epoch 118:
Train loss: 82.63285626471043, Train accuracy: 77.19%
Validation loss: 0.87056843, Validation accuracy: 63.97%

Epoch 119:
Train loss: 82.23556263744831, Train accuracy: 77.38%
Validation loss: 0.90766458, Validation accuracy: 61.00%

Epoch 120:
Train loss: 83.17733739316463, Train accuracy: 76.98%
Validation loss: 0.85833261, Validation accuracy: 62.84%

Epoch 121:
Train loss: 82.48346854746342, Train accuracy: 77.22%
Validation loss: 0.82601221, Validation accuracy: 63.87%

Epoch 122:
Train loss: 84.2340731471777, Train accuracy: 76.95%
Validation loss: 0.80315320, Validation accuracy: 64.33%

Epoch 123:
Train loss: 82.38418617844582, Train accuracy: 77.48%
Validation loss: 0.81024953, Validation accuracy: 63.62%

Epoch 124:
Train loss: 82.61791820824146, Train accuracy: 77.54%
Validation loss: 0.85065120, Validation accuracy: 60.86%

Epoch 125:
Train loss: 84.25584715604782, Train accuracy: 76.87%
Validation loss: 0.93960699, Validation accuracy: 61.22%

Epoch 126:
Train loss: 81.5576134622097, Train accuracy: 77.49%
Validation loss: 0.89574565, Validation accuracy: 61.37%

Epoch 127:
Train loss: 81.93572431057692, Train accuracy: 77.49%
Validation loss: 0.85740920, Validation accuracy: 61.54%

Epoch 128:
Train loss: 80.4118178486824, Train accuracy: 77.94%
Validation loss: 0.95213185, Validation accuracy: 61.54%

Epoch 129:
Train loss: 80.83019641041756, Train accuracy: 78.07%
Validation loss: 0.89333736, Validation accuracy: 62.20%

Epoch 130:
Train loss: 81.01710495352745, Train accuracy: 77.84%
Validation loss: 0.92750035, Validation accuracy: 61.13%

Epoch 131:
Train loss: 81.41367541253567, Train accuracy: 77.74%
Validation loss: 0.96288697, Validation accuracy: 61.03%

Epoch 132:
Train loss: 80.7163021415472, Train accuracy: 77.64%
Validation loss: 0.83319412, Validation accuracy: 62.82%

Epoch 133:
Train loss: 80.47476309537888, Train accuracy: 78.16%
Validation loss: 0.88412105, Validation accuracy: 61.54%

Epoch 134:
Train loss: 79.03802996128798, Train accuracy: 78.70%
Validation loss: 0.87211549, Validation accuracy: 62.57%

Epoch 135:
Train loss: 79.58140673488379, Train accuracy: 78.33%
Validation loss: 0.77858412, Validation accuracy: 64.48%

Epoch 136:
Train loss: 78.97485605627298, Train accuracy: 78.63%
Validation loss: 0.83103295, Validation accuracy: 62.69%

Epoch 137:
Train loss: 80.32298094034195, Train accuracy: 78.40%
Validation loss: 0.83092279, Validation accuracy: 63.06%

Epoch 138:
Train loss: 79.76009928435087, Train accuracy: 78.45%
Validation loss: 0.88131258, Validation accuracy: 62.74%

Epoch 139:
Train loss: 79.87181838601828, Train accuracy: 78.33%
Validation loss: 0.86701749, Validation accuracy: 63.01%

Epoch 140:
Train loss: 79.14102477580309, Train accuracy: 78.49%
Validation loss: 0.80572665, Validation accuracy: 62.94%

Epoch 141:
Train loss: 80.14948266744614, Train accuracy: 78.05%
Validation loss: 0.85189989, Validation accuracy: 59.83%

Epoch 142:
Train loss: 79.64087118208408, Train accuracy: 78.31%
Validation loss: 0.88671087, Validation accuracy: 61.86%

Epoch 143:
Train loss: 79.68530815839767, Train accuracy: 78.48%
Validation loss: 0.88797557, Validation accuracy: 60.56%

Epoch 144:
Train loss: 79.55032539367676, Train accuracy: 78.45%
Validation loss: 0.87436590, Validation accuracy: 61.42%

Epoch 145:
Train loss: 80.76893322169781, Train accuracy: 77.73%
Validation loss: 0.87798505, Validation accuracy: 61.22%

Epoch 146:
Train loss: 80.99634605646133, Train accuracy: 77.83%
Validation loss: 0.92969964, Validation accuracy: 61.20%

Epoch 147:
Train loss: 80.45237195491791, Train accuracy: 77.90%
Validation loss: 0.87712038, Validation accuracy: 61.86%

Epoch 148:
Train loss: 79.20750214159489, Train accuracy: 78.28%
Validation loss: 0.88069652, Validation accuracy: 61.37%

Epoch 149:
Train loss: 79.84993623197079, Train accuracy: 78.15%
Validation loss: 0.83726426, Validation accuracy: 62.13%

Epoch 150:
Train loss: 79.40920224785805, Train accuracy: 78.39%
Validation loss: 0.96670942, Validation accuracy: 60.44%

Epoch 151:
Train loss: 79.95442485064268, Train accuracy: 78.08%
Validation loss: 0.87855956, Validation accuracy: 61.10%

Epoch 152:
Train loss: 80.27177917212248, Train accuracy: 77.92%
Validation loss: 0.93703172, Validation accuracy: 60.05%

Epoch 153:
Train loss: 80.02470774948597, Train accuracy: 78.36%
Validation loss: 0.91843312, Validation accuracy: 61.00%

Epoch 154:
Train loss: 81.67303451895714, Train accuracy: 77.31%
Validation loss: 0.90838475, Validation accuracy: 60.78%

Epoch 155:
Train loss: 82.5789654403925, Train accuracy: 77.06%
Validation loss: 0.84472632, Validation accuracy: 62.64%

Epoch 156:
Train loss: 80.56506446003914, Train accuracy: 77.85%
Validation loss: 0.95176685, Validation accuracy: 60.17%

Epoch 157:
Train loss: 81.35521154105663, Train accuracy: 77.32%
Validation loss: 0.81395462, Validation accuracy: 63.26%

Epoch 158:
Train loss: 81.32347769290209, Train accuracy: 77.44%
Validation loss: 0.90628110, Validation accuracy: 60.78%

Epoch 159:
Train loss: 81.03820259869099, Train accuracy: 77.66%
Validation loss: 0.85749740, Validation accuracy: 62.55%

Epoch 160:
Train loss: 79.33452928066254, Train accuracy: 78.47%
Validation loss: 0.90811119, Validation accuracy: 61.03%
"""

# Extract data using regular expressions
epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

pattern = re.compile(r"Epoch (\d+):\nTrain loss: ([\d\.e\-]+), Train accuracy: ([\d\.]+)%\nValidation loss: ([\d\.]+), Validation accuracy: ([\d\.]+)%")
matches = pattern.findall(data)

for match in matches:
    epochs.append(int(match[0]))
    train_loss.append(float(match[1]))
    val_loss.append(float(match[3]))
    train_acc.append(float(match[2]))
    val_acc.append(float(match[4]))

# # Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'o-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, 'o-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'o-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('SW-MHSA Model Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()