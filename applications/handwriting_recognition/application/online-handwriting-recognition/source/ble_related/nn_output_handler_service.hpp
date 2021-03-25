#pragma once

#include "platform/Callback.h"
#include "events/EventQueue.h"
#include "ble/BLE.h"
#include "gatt_server_process.h"

#include "utils_ohw.hpp"
#include <stdint.h>
#include <sys/types.h>

#define DEBUG_BLE_SERVICE 0
#define WRITE_CHARACTERISTICS_IMPLEMENTED 0
#define N_LABELS 16 // TODO SET AS TEMPLATE



using mbed::callback;
using namespace std::literals::chrono_literals;

/**
 * Neural network output handler service that sends the average scores and prediction.
 *
 * A client can subscribe to updates of the nn_output_handler characteristics and get
 * notified when one of the values is changed. 
 *
 */
class NNOutputHandlerService : public ble::GattServer::EventHandler {
public:
    NNOutputHandlerService() :
        _prediction_char("7f028bda-2032-4982-b1b7-7a121e70c6bc", 0, GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY),
        _avg_scores_char("d10b1962-4a92-4432-90ac-ebe706701d33", 0, GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY),
        _nn_output_handler_service(
            /* uuid */ "a29cee85-4fd7-4118-be91-f77d40db9362",
            /* characteristics */ _nn_output_handler_characteristics,
            /* numCharacteristics */ sizeof(_nn_output_handler_characteristics) /
                                     sizeof(_nn_output_handler_characteristics[0])
        )
    {
        /* update internal pointers (value, descriptors and characteristics array) */
        _nn_output_handler_characteristics[0] = &_prediction_char;
        _nn_output_handler_characteristics[1] = &_avg_scores_char;

        /* setup authorization handlers */ /* For the moment there are no client write characteristics */
        //_prediction_char.setWriteAuthorizationCallback(this, &NNOutputHandlerService::authorize_client_write);
        //_avg_scores_char.setWriteAuthorizationCallback(this, &NNOutputHandlerService::authorize_client_write);
    }

    void start(BLE &ble, events::EventQueue &event_queue){
        _server = &ble.gattServer();
        _event_queue = &event_queue;

        // Register service
        ble_error_t err = _server->addService(_nn_output_handler_service);

        if (err) {
            pc.printf("Error %u during NNOutputHandlerService registration.\r\n", err);
            return;
        }

        /* register handlers */
        _server->setEventHandler(this);

        pc.printf("NNOutputHandlerService service registered\r\n");
        pc.printf("service handle: %u\r\n", _nn_output_handler_service.getHandle());
        pc.printf("prediction characteristic value handle %u\r\n", _prediction_char.getValueHandle());
        pc.printf("avg_scores characteristic value handle %u\r\n", _avg_scores_char.getValueHandle());

        //_event_queue->call_every(1000ms, callback(this, &NNOutputHandlerService::get_new_prediction));
    }

    /**
    * Queues a new prediction to be sent over BLE
    * @param[in] prediction The label index when a detection is made
    */
    void set_new_prediction(uint8_t prediction){
        _event_queue->call(callback(this, &NNOutputHandlerService::send_new_prediction), prediction);
    }

    /**
    * Queues a new avg_scores to be sent over BLE
    * @param[in] *avg_scores uint8_t array containing the scores for each label
    */
    void set_new_scores(uint8_t *avg_scores){
        _event_queue->call(callback(this, &NNOutputHandlerService::send_new_scores), avg_scores);
    }

    /* GattServer::EventHandler */
/* Only for debugging in this case */
#if DEBUG_BLE_SERVICE
private:
    /**
     * Handler called when a notification or an indication has been sent.
     */
    void onDataSent(const GattDataSentCallbackParams &params) override
    {
        pc.printf("sent updates\r\n");
    }

    /**
     * Handler called after an attribute has been written.
     */
    void onDataWritten(const GattWriteCallbackParams &params) override
    {
        pc.printf("data written:\r\n");
        pc.printf("connection handle: %u\r\n", params.connHandle);
        pc.printf("attribute handle: %u", params.handle);

        if (params.handle == _prediction_char.getValueHandle()) {
            pc.printf(" (prediction characteristic)\r\n");
        } else if (params.handle == _avg_scores_char.getValueHandle()) {
            pc.printf(" (avg_scores characteristic)\r\n");
        } else {
            pc.printf("\r\n");
        }
        pc.printf("write operation: %u\r\n", params.writeOp);
        pc.printf("offset: %u\r\n", params.offset);
        pc.printf("length: %u\r\n", params.len);
        pc.printf("data: ");

        for (size_t i = 0; i < params.len; ++i) {
            pc.printf("%02X", params.data[i]);
        }

        pc.printf("\r\n");
    }

    /**
     * Handler called after an attribute has been read.
     */
    void onDataRead(const GattReadCallbackParams &params) override
    {
        pc.printf("data read:\r\n");
        pc.printf("connection handle: %u\r\n", params.connHandle);
        pc.printf("attribute handle: %u", params.handle);

        if (params.handle == _prediction_char.getValueHandle()) {
            pc.printf(" (prediction characteristic)\r\n");
        } else if (params.handle == _avg_scores_char.getValueHandle()) {
            pc.printf(" (avg_scores characteristic)\r\n");
        } else {
            pc.printf("\r\n");
        }
    }

    /**
     * Handler called after a client has subscribed to notification or indication.
     *
     * @param handle Handle of the characteristic value affected by the change.
     */
    void onUpdatesEnabled(const GattUpdatesEnabledCallbackParams &params) override
    {
        pc.printf("update enabled on handle %d\r\n", params.attHandle);
    }

    /**
     * Handler called after a client has cancelled his subscription from
     * notification or indication.
     *
     * @param handle Handle of the characteristic value affected by the change.
     */
    void onUpdatesDisabled(const GattUpdatesDisabledCallbackParams &params) override
    {
        pc.printf("update disabled on handle %d\r\n", params.attHandle);
    }

    /**
     * Handler called when an indication confirmation has been received.
     *
     * @param handle Handle of the characteristic value that has emitted the
     * indication.
     */
    void onConfirmationReceived(const GattConfirmationReceivedCallbackParams &params) override
    {
        pc.printf("confirmation received on handle %d\r\n", params.attHandle);
    }
#endif

/* For the moment there are no client write characteristics */
#if WRITE_CHARACTERISTICS_IMPLEMENTED
private:
    /**
     * Handler called when a write request is received.
     *
     * This handler verify that the value submitted by the client is valid before
     * authorizing the operation.
     */
    void authorize_client_write(GattWriteAuthCallbackParams *e)
    {
        pc.printf("characteristic %u write authorization\r\n", e->handle);

        GattAttribute::Handle_t target_handle = e->handle;
        uint16_t char_length;

        // Get corresponding correct data length
        if(target_handle == _prediction_char.getValueHandle()){
            char_length = _prediction_char.getValueAttribute().getLength();
        } else if (target_handle == _avg_scores_char.getValueHandle()){
            char_length = _avg_scores_char.getValueAttribute().getLength();
        } else char_length = 0;

        if (e->offset != 0) {
            pc.printf("Error invalid offset\r\n");
            e->authorizationReply = AUTH_CALLBACK_REPLY_ATTERR_INVALID_OFFSET;
            return;
        }

        if (e->len != char_length) {
            pc.printf("Error invalid len\r\n");
            e->authorizationReply = AUTH_CALLBACK_REPLY_ATTERR_INVALID_ATT_VAL_LENGTH;
            return;
        }

        /* Data Restrictions */
        /*
        if (e->data[foo] == bar) {
            pc.printf("Error invalid data\r\n");
            e->authorizationReply = AUTH_CALLBACK_REPLY_ATTERR_WRITE_NOT_PERMITTED;
            return;
        }
        */

        e->authorizationReply = AUTH_CALLBACK_REPLY_SUCCESS;
    }

    
#endif

private:

    /**
     * Sends a prediction over BLE
     * @param[in] prediction The label index when a detection is made
     */
    void send_new_prediction(uint8_t prediction){
        ble_error_t err;

        // prediction characteristic
        err = characteristic_write(_prediction_char.getValueHandle(), prediction, sizeof(unsigned char) );
        if (err) {
            pc.printf("write of the prediction value returned error %u\r\n", err);
            return;
        }
    }

    /**
    * Sends a new avg_scores to be sent over BLE
    * @param[in] *avg_scores uint8_t array containing the scores for each label
    */
    void send_new_scores(uint8_t *avg_scores){
        ble_error_t err;
        // avg_scores characteristic
        err = characteristic_write(_avg_scores_char.getValueHandle(), *avg_scores, sizeof(uint8_t)*N_LABELS);
        if (err) {
            pc.printf("write of the avg_scores value returned error %u\r\n", err);
            return;
        }

    }

    /**
    * Get the value of this characteristic.
    *
    * @param[in] GattAttribute::Handle_t handle of the characteristic to be read from.
    * @param[out] dst Variable that will receive the characteristic value.
    *
    * @return BLE_ERROR_NONE in case of success or an appropriate error code.
    */
    template<typename T>
    ble_error_t characteristic_read(GattAttribute::Handle_t handle, T& dst){
        uint16_t value_length = sizeof(dst);
        return _server->read(handle, (uint8_t *)&dst, &value_length);
    }

    /**
    * Assign a new value to this characteristic.
    *
    * @param[in] GattAttribute::Handle_t handle of the characteristic to be written to.
    * @param[in] value The new value to set.
    * @param[in] size The number of bytes to write
    * @param[in] local_only Flag that determine if the change should be kept
    *   locally or forwarded to subscribed clients.
    */
    template<typename T>
    ble_error_t characteristic_write(GattAttribute::Handle_t handle, const T& value, size_t size, bool local_only = false){
        uint8_t *p = (uint8_t *) &value;
        return _server->write(handle, p, size, local_only);
    }
    

private:
    GattServer *_server = nullptr;
    events::EventQueue *_event_queue = nullptr;
    
    // Characteristics
    ReadOnlyGattCharacteristic<unsigned char> _prediction_char;
    ReadOnlyArrayGattCharacteristic<uint8_t, N_LABELS> _avg_scores_char;

    GattService _nn_output_handler_service;
    GattCharacteristic* _nn_output_handler_characteristics[2];

};
