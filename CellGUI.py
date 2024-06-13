import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QDialog, QFormLayout


class CellInputDialog(QDialog):
    def __init__(self, cell_number):
        super().__init__()
        self.cell_number = cell_number
        self.setWindowTitle(f'Cell {cell_number} Information')
        self.setGeometry(100, 100, 400, 300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.specie_edit = QLineEdit()
        self.flow_behavior_edit = QLineEdit()
        self.input_connections_edit = QLineEdit()
        self.output_connections_edit = QLineEdit()

        self.specie_label = QLabel("Specie:")
        self.flow_behavior_label = QLabel("Flow Behavior:")
        self.input_connections_label = QLabel("Input Connections:")
        self.output_connections_label = QLabel("Output Connections:")

        layout.addWidget(self.specie_label)
        layout.addWidget(self.specie_edit)
        layout.addWidget(self.flow_behavior_label)
        layout.addWidget(self.flow_behavior_edit)
        layout.addWidget(self.input_connections_label)
        layout.addWidget(self.input_connections_edit)
        layout.addWidget(self.output_connections_label)
        layout.addWidget(self.output_connections_edit)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.accept)

        layout.addWidget(submit_button)

        self.setLayout(layout)

    def get_cell_info(self):
        self.exec_()
        return (
            self.specie_edit.text(),
            self.flow_behavior_edit.text(),
            self.input_connections_edit.text(),
            self.output_connections_edit.text()
        )


class CellInfoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Experiment Information')
        self.setGeometry(100, 100, 400, 300)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.num_cells_label = QLabel('Number of Cells:')
        layout.addWidget(self.num_cells_label)

        self.num_cells_input = QLineEdit()
        layout.addWidget(self.num_cells_input)

        self.experiment_duration_label = QLabel('Experiment Duration:')
        layout.addWidget(self.experiment_duration_label)

        self.experiment_duration_input = QLineEdit()
        layout.addWidget(self.experiment_duration_input)

        self.num_medias_label = QLabel('Number of Medias:')
        layout.addWidget(self.num_medias_label)

        self.num_medias_input = QLineEdit()
        layout.addWidget(self.num_medias_input)

        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.get_cell_info)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def get_cell_info(self):
        num_cells = int(self.num_cells_input.text())
        experiment_duration = self.experiment_duration_input.text()
        num_medias = int(self.num_medias_input.text())

        cell_info = {}
        for i in range(1, num_cells + 1):
            dialog = CellInputDialog(i)
            specie, flow_behavior, input_connections, output_connections = dialog.get_cell_info()
            cell_info[f"Cell {i}"] = {
                "Specie": specie,
                "Flow Behavior": flow_behavior,
                "Input Connections": input_connections,
                "Output Connections": output_connections
            }

        self.show_message(experiment_duration, num_medias, cell_info)

    def show_message(self, experiment_duration, num_medias, cell_info):
        message = f'Experiment Duration: {experiment_duration}\nNumber of Medias: {num_medias}\n\n'
        message += 'Cell Information:\n'
        for cell, info in cell_info.items():
            message += f'\n{cell}:\n'
            for key, value in info.items():
                message += f'{key}: {value}\n'

        QMessageBox.information(self, 'Cell Information', message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CellInfoWindow()
    window.show()
    sys.exit(app.exec_())
