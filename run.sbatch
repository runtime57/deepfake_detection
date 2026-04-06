#!/bin/bash
#SBATCH --job-name=train_amsdf_full        # Название задачи
#SBATCH --error=task-%j.err                # Файл для вывода ошибок
#SBATCH --output=task-%j.log               # Файл для вывода результатов
#SBATCH --time=100:00:00                   # Максимальное время выполнения
#SBATCH --nodes=1                          # Требуемое кол-во узлов
#SBATCH --gpus=1                           # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1                  # Требуемое кол-во CPU
#SBATCH --mail-user=ialarin@edu.hse.ru     # Укажи�~Bе ва�~H email дл�~O о�~Bп�~@авки �~Cведомлений
#SBATCH --mail-type=END,FAIL               # Соб�~K�~Bи�~O, �~B�~@еб�~C�~N�~Iие �~Cведомлени�~O

python /home/ialarin/deepfake_detection/train.py --cn=baseline            # Выполнение расчёта
#srun {options} {command}                                                 # Выполнение MPI расчёта