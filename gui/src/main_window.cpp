#include "main_window.h"

#include <QWidget>
#include <QTextEdit>
#include <QHBoxLayout>

#include "ptx_generator.h"

MainWindow::MainWindow()
{
  QTextEdit *edit = new QTextEdit();
  QTextEdit *ptx = new QTextEdit();

  edit->setAcceptRichText(false);
  edit->setPlainText("__global__ void kernel(int *data)\n"
                     "{\n"
                     "    data[0] = 42;\n"
                     "}");

  QHBoxLayout *layout = new QHBoxLayout();
  layout->addWidget(edit);
  layout->addWidget(ptx);

  QWidget *central_widget = new QWidget();
  central_widget->setLayout(layout);

  setCentralWidget(central_widget);

  std::string cuda_source = edit->toPlainText().toStdString();

  PTXGenerator generator;
  std::optional<PTXCode> ptx_code = generator.gen(cuda_source.c_str(), { "-lineinfo", "--gpu-architecture=compute_86" });

  if (ptx_code)
  {
    ptx->setPlainText(ptx_code->get_ptx());
  }
  else
  {
    ptx->setPlainText("Compilation error");
  }

}