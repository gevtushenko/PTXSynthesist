#include "main_window.h"

#include <QTimer>
#include <QWidget>
#include <QTextEdit>
#include <QHBoxLayout>

#include "ptx_generator.h"

MainWindow::MainWindow()
  : cuda(new QTextEdit())
  , ptx(new QTextEdit())
  , timer(new QTimer())
{
  cuda->setAcceptRichText(false);
  cuda->setPlainText("__global__ void kernel(int *data)\n"
                     "{\n"
                     "    data[0] = 42;\n"
                     "}");

  QHBoxLayout *layout = new QHBoxLayout();
  layout->addWidget(cuda);
  layout->addWidget(ptx);

  QWidget *central_widget = new QWidget();
  central_widget->setLayout(layout);

  setCentralWidget(central_widget);

  timer->setSingleShot(true);

  QObject::connect(cuda->document(), &QTextDocument::contentsChanged, this, &MainWindow::reset_timer);
  QObject::connect(timer, &QTimer::timeout, this, &MainWindow::regen_ptx);

  reset_timer();
}

void MainWindow::reset_timer()
{
  timer->start(500);
}

#include <iostream>

void MainWindow::regen_ptx()
{
  std::string cuda_source = cuda->toPlainText().toStdString();

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
