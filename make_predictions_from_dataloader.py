import torch
import numpy as np


def make_predictions_from_dataloader(model, unshuffled_dataloader, dframe, device):
    model.eval()
    predictions, actuals = [], []
    firstloop = True
    for x, y in unshuffled_dataloader:
        with torch.no_grad():
            if firstloop:
                # dans cette base de test qui contient des séquences
                # on s'intéresse à la première séquence ici avec le 'if firstloop'
                # on s'intéresse à la dernière donnée de cette première séquence
                # pour récupérer la date yday et le nombre de secondes sec
                # ça permettra de plot le predicted vs actual avec les bonnes dates correspondantes aux données
                # sachant que pour une target datée à K, la séquence d'entrée s'étend de K-pw à K-1
                # pw étant la taille prédéfinie de la séquence
                # on cherche ici à récupérer cette notion de date K

                seq = x[0]
                yday_cos = seq[-1][0].item()
                yday_sin = seq[-1][1].item()
                sec_cos = seq[-1][2].item()
                sec_sin = seq[-1][3].item()
                firstloop = False

            x = x.to(device)
            y = y.squeeze().to(device)

            p = model(x).squeeze()
            predictions = np.append(predictions, p)
            # print(predictions.shape)
            actuals = np.append(actuals, y)
            # print(actuals.shape)
    # on cherche comme expliqué, la ligne du dataframe correspondante à cette date

    k_row = dframe.loc[(dframe['cyclic_yday_cos'] == yday_cos) &
                       (dframe['cyclic_yday_sin'] == yday_sin) &
                       (dframe['cyclic_sec_cos'] == sec_cos) &
                       (dframe['cyclic_sec_sin'] == sec_sin)]

    k_idx = k_row.index.values[0] + 1
    return predictions, actuals, k_idx
